import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper

class TabularDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]


class MLPNet(nn.Module):
    def __init__(self, cat_dims, num_dim, emb_dims, layers, dropout=0.1, activation="relu"):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c_dim, embedding_dim=e_dim)
            for c_dim, e_dim in zip(cat_dims, emb_dims)
        ])
        tot_emb_dim = sum(emb_dims)
        in_dim = tot_emb_dim + num_dim
        
        act_fn = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "selu": nn.SELU,
            "leaky_relu": nn.LeakyReLU
        }.get(activation, nn.ReLU)()
        
        net_layers = []
        curr_dim = in_dim
        for l_dim in layers:
            net_layers.append(nn.Linear(curr_dim, l_dim))
            net_layers.append(nn.BatchNorm1d(l_dim))
            net_layers.append(act_fn)
            net_layers.append(nn.Dropout(dropout))
            curr_dim = l_dim
        net_layers.append(nn.Linear(curr_dim, 1))
        
        self.mlp = nn.Sequential(*net_layers)

    def forward(self, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_num], dim=1)
        return self.mlp(x).squeeze(1)


class MLPWrapper(BaseModelWrapper):
    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("mlp", config, tune)
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split_features(self, X):
        if isinstance(X, np.ndarray):
            df_temp = pd.DataFrame(X, columns=self.cat_cols + self.num_cols)
        else:
            df_temp = X
        X_cat = df_temp[self.cat_cols].values.astype(np.int64)
        X_num = df_temp[self.num_cols].values.astype(np.float32)
        return X_cat, X_num

    def fit(self, X_train, y_train, X_val, y_val):
        X_train_cat, X_train_num = self._split_features(X_train)
        X_val_cat, X_val_num = self._split_features(X_val)
        
        if len(self.cat_cols) == 4:
            cat_dims = [887, 5, 26, 4]
        else:
            cat_dims = [887, 5, 26]
        num_dim = X_train_num.shape[1]
        
        if self.tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                # 1. Network architecture parameters
                num_layers = trial.suggest_int("num_layers", 1, 4)
                layers = []
                for i in range(num_layers):
                    layers.append(trial.suggest_categorical(f"layer_{i}_size", [32, 64, 128, 256, 512]))
                
                # 2. Embedding dimension scaling factor
                emb_scale = trial.suggest_float("emb_scale", 0.5, 2.0)
                if len(self.cat_cols) == 4:
                    emb_dims = [
                        max(2, int(16 * emb_scale)),
                        max(2, int(4 * emb_scale)),
                        max(2, int(8 * emb_scale)),
                        max(2, int(4 * emb_scale))
                    ]
                else:
                    emb_dims = [
                        max(2, int(16 * emb_scale)),
                        max(2, int(4 * emb_scale)),
                        max(2, int(8 * emb_scale))
                    ]
                
                # 3. Regularization & Optimization parameters
                dropout = trial.suggest_float("dropout", 0.0, 0.6)
                activation = trial.suggest_categorical("activation", ["relu", "gelu", "selu", "leaky_relu"])
                optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd", "rmsprop"])
                lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
                weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
                
                # Setup dataloaders
                train_dataset = TabularDataset(X_train_cat, X_train_num, y_train.values if isinstance(y_train, pd.Series) else y_train)
                val_dataset = TabularDataset(X_val_cat, X_val_num, y_val.values if isinstance(y_val, pd.Series) else y_val)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize model for trial
                trial_model = MLPNet(cat_dims, num_dim, emb_dims, layers, dropout, activation).to(self.device)
                criterion = nn.BCEWithLogitsLoss()
                
                # Optimizer mapping
                if optimizer_name == "adam":
                    opt = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                elif optimizer_name == "adamw":
                    opt = optim.AdamW(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                elif optimizer_name == "sgd":
                    opt = optim.SGD(trial_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
                else: # rmsprop
                    opt = optim.RMSprop(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
                
                epochs_to_run = 2 if self.config.get("epochs", 20) <= 2 else 5
                
                for epoch in range(epochs_to_run):
                    trial_model.train()
                    for x_cat, x_num, y in train_loader:
                        x_cat, x_num, y = x_cat.to(self.device), x_num.to(self.device), y.to(self.device)
                        opt.zero_grad()
                        outputs = trial_model(x_cat, x_num)
                        loss = criterion(outputs, y)
                        loss.backward()
                        opt.step()
                        
                trial_model.eval()
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for x_cat, x_num, y in val_loader:
                        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
                        outputs = torch.sigmoid(trial_model(x_cat, x_num))
                        val_preds.extend(outputs.cpu().numpy())
                        val_targets.extend(y.numpy())
                        
                return roc_auc_score(val_targets, val_preds)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)  # 30 trials for MLP
            print(f"[{self.model_name}] Best Params: {study.best_params}")
            self.config.update(study.best_params)
            
        # Reconstruct layers and emb_dims from config
        num_layers = self.config.get("num_layers", 2)
        layers = []
        for i in range(num_layers):
            layers.append(self.config.get(f"layer_{i}_size", 128 if i == 0 else 64))
            
        emb_scale = self.config.get("emb_scale", 1.0)
        if len(self.cat_cols) == 4:
            emb_dims = [
                max(2, int(16 * emb_scale)),
                max(2, int(4 * emb_scale)),
                max(2, int(8 * emb_scale)),
                max(2, int(4 * emb_scale))
            ]
        else:
            emb_dims = [
                max(2, int(16 * emb_scale)),
                max(2, int(4 * emb_scale)),
                max(2, int(8 * emb_scale))
            ]
        
        lr = self.config.get("lr", 1e-3)
        batch_size = self.config.get("batch_size", 512)
        epochs = self.config.get("epochs", 20)
        patience = self.config.get("patience", 3)
        dropout = self.config.get("dropout", 0.1)
        activation = self.config.get("activation", "relu")
        optimizer_name = self.config.get("optimizer", "adam")
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        train_dataset = TabularDataset(X_train_cat, X_train_num, y_train.values if isinstance(y_train, pd.Series) else y_train)
        val_dataset = TabularDataset(X_val_cat, X_val_num, y_val.values if isinstance(y_val, pd.Series) else y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.model = MLPNet(cat_dims, num_dim, emb_dims, layers, dropout, activation).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else: # rmsprop
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        best_auc = 0.0
        best_state = None
        patience_counter = 0
        
        print(f"[{self.model_name}] Training on {self.device} with config: {self.config}...")
        for epoch in range(epochs):
            self.model.train()
            for x_cat, x_num, y in train_loader:
                x_cat, x_num, y = x_cat.to(self.device), x_num.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_cat, x_num)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for x_cat, x_num, y in val_loader:
                    x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
                    outputs = torch.sigmoid(self.model(x_cat, x_num))
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(y.numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best Val AUC: {best_auc:.5f}")
                break
                
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
    def predict_proba(self, X):
        self.model.eval()
        X_cat, X_num = self._split_features(X)
        dataset = TabularDataset(X_cat, X_num)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for x_cat, x_num in loader:
                x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
                outputs = torch.sigmoid(self.model(x_cat, x_num))
                preds.extend(outputs.cpu().numpy())
        return np.array(preds)
        
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'cat_cols': self.cat_cols,
            'num_cols': self.num_cols
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.cat_cols = checkpoint['cat_cols']
        self.num_cols = checkpoint['num_cols']
        
        if len(self.cat_cols) == 4:
            cat_dims = [887, 5, 26, 4]
        else:
            cat_dims = [887, 5, 26]
        num_dim = len(self.num_cols)
        
        num_layers = self.config.get("num_layers", 2)
        layers = []
        for i in range(num_layers):
            layers.append(self.config.get(f"layer_{i}_size", 128 if i == 0 else 64))
            
        emb_scale = self.config.get("emb_scale", 1.0)
        if len(self.cat_cols) == 4:
            emb_dims = [
                max(2, int(16 * emb_scale)),
                max(2, int(4 * emb_scale)),
                max(2, int(8 * emb_scale)),
                max(2, int(4 * emb_scale))
            ]
        else:
            emb_dims = [
                max(2, int(16 * emb_scale)),
                max(2, int(4 * emb_scale)),
                max(2, int(8 * emb_scale))
            ]
        
        dropout = self.config.get("dropout", 0.1)
        activation = self.config.get("activation", "relu")
        
        self.model = MLPNet(cat_dims, num_dim, emb_dims, layers, dropout, activation).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

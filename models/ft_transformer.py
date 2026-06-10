"""FT-Transformer wrapper.

Implements the Feature Tokenizer + Transformer architecture (Gorishniy et al.,
2021): every feature (numeric or categorical) becomes a d_token embedding, a
learnable [CLS] token is prepended, a Transformer encoder attends over the tokens,
and the [CLS] representation is classified. Training scaffold (TabularDataset,
early stopping) matches the MLP wrapper for repo consistency.
"""
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper

CAT_DIMS_4 = [887, 5, 26, 4]   # Driver, Compound, Race, Year (FE39)
CAT_DIMS_3 = [887, 5, 26]


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


class FTTransformerNet(nn.Module):
    def __init__(self, cat_dims, num_dim, d_token=192, depth=3, n_heads=8,
                 ffn_factor=1.33, dropout=0.1):
        super().__init__()
        self.num_dim = num_dim
        # numeric feature i -> x_i * W_i + b_i  (a d_token vector)
        self.num_weight = nn.Parameter(torch.empty(num_dim, d_token))
        self.num_bias = nn.Parameter(torch.empty(num_dim, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        nn.init.normal_(self.num_bias, std=0.02)
        self.cat_embs = nn.ModuleList(nn.Embedding(c, d_token) for c in cat_dims)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=int(d_token * ffn_factor),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(inplace=True), nn.Linear(d_token, 1))

    def forward(self, x_cat, x_num):
        num_tok = x_num.unsqueeze(-1) * self.num_weight + self.num_bias   # (B,n_num,d)
        cat_tok = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
        tok = torch.cat([num_tok, cat_tok], dim=1)
        cls = self.cls.expand(tok.size(0), -1, -1)
        z = self.encoder(torch.cat([cls, tok], dim=1))
        return self.head(z[:, 0]).squeeze(1)


class FTTransformerWrapper(BaseModelWrapper):
    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("ft_transformer", config, tune)
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _split(self, X):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.cat_cols + self.num_cols)
        return (df[self.cat_cols].values.astype(np.int64),
                df[self.num_cols].values.astype(np.float32))

    def _build(self, num_dim):
        cat_dims = CAT_DIMS_4 if len(self.cat_cols) == 4 else CAT_DIMS_3
        return FTTransformerNet(
            cat_dims, num_dim,
            d_token=self.config.get("d_token", 192),
            depth=self.config.get("depth", 3),
            n_heads=self.config.get("n_heads", 8),
            ffn_factor=self.config.get("ffn_factor", 1.33),
            dropout=self.config.get("dropout", 0.1)).to(self.device)

    def fit(self, X_train, y_train, X_val, y_val):
        Xtc, Xtn = self._split(X_train)
        Xvc, Xvn = self._split(X_val)
        yt = y_train.values if isinstance(y_train, pd.Series) else y_train
        yv = y_val.values if isinstance(y_val, pd.Series) else y_val

        lr = self.config.get("lr", 1e-4)
        batch_size = self.config.get("batch_size", 1024)
        epochs = self.config.get("epochs", 30)
        patience = self.config.get("patience", 5)
        weight_decay = self.config.get("weight_decay", 1e-5)

        train_loader = DataLoader(TabularDataset(Xtc, Xtn, yt), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TabularDataset(Xvc, Xvn, yv), batch_size=2048, shuffle=False)

        self.model = self._build(Xtn.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_auc, best_state, patience_ctr = 0.0, None, 0
        print(f"[{self.model_name}] Training on {self.device} ...")
        for epoch in range(epochs):
            self.model.train()
            for x_cat, x_num, y in train_loader:
                x_cat, x_num, y = x_cat.to(self.device), x_num.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x_cat, x_num), y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x_cat, x_num, y in val_loader:
                    out = torch.sigmoid(self.model(x_cat.to(self.device), x_num.to(self.device)))
                    preds.extend(out.cpu().numpy()); targets.extend(y.numpy())
            val_auc = roc_auc_score(targets, preds)
            if val_auc > best_auc:
                best_auc, best_state, patience_ctr = val_auc, self.model.state_dict().copy(), 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Val AUC: {best_auc:.5f}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X):
        self.model.eval()
        Xc, Xn = self._split(X)
        loader = DataLoader(TabularDataset(Xc, Xn), batch_size=2048, shuffle=False)
        preds = []
        with torch.no_grad():
            for x_cat, x_num in loader:
                out = torch.sigmoid(self.model(x_cat.to(self.device), x_num.to(self.device)))
                preds.extend(out.cpu().numpy())
        return np.array(preds)

    def save(self, path):
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config,
                    "cat_cols": self.cat_cols, "num_cols": self.num_cols}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt["config"]; self.cat_cols = ckpt["cat_cols"]; self.num_cols = ckpt["num_cols"]
        self.model = self._build(len(self.num_cols))
        self.model.load_state_dict(ckpt["model_state_dict"])

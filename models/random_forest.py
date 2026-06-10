import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper

class RandomForestWrapper(BaseModelWrapper):
    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("random_forest", config, tune)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
                
                max_depth_choice = trial.suggest_categorical("max_depth_choice", ["None", "5", "10", "15", "20", "25", "30"])
                max_depth = None if max_depth_choice == "None" else int(max_depth_choice)
                
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
                
                max_features_choice = trial.suggest_categorical("max_features_choice", ["sqrt", "log2", "0.2", "0.4", "0.6", "0.8", "None"])
                if max_features_choice == "sqrt":
                    max_features = "sqrt"
                elif max_features_choice == "log2":
                    max_features = "log2"
                elif max_features_choice == "None":
                    max_features = None
                else:
                    max_features = float(max_features_choice)
                    
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
                class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
                
                bootstrap = trial.suggest_categorical("bootstrap", [True, False])
                if bootstrap:
                    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
                else:
                    max_samples = None
                    
                ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.02)
                min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    criterion=criterion,
                    class_weight=class_weight,
                    bootstrap=bootstrap,
                    max_samples=max_samples,
                    ccp_alpha=ccp_alpha,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, preds)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)  # Increased trials to 40
            print(f"[{self.model_name}] Best Params: {study.best_params}")
            self.config.update(study.best_params)
            
        # Reconstruction using best parameters
        max_depth_choice = self.config.get("max_depth_choice", "10")
        max_depth = None if max_depth_choice == "None" else int(max_depth_choice)
        
        max_features_choice = self.config.get("max_features_choice", "sqrt")
        if max_features_choice in ["sqrt", "log2"]:
            max_features = max_features_choice
        elif max_features_choice == "None":
            max_features = None
        else:
            max_features = float(max_features_choice)
            
        bootstrap = self.config.get("bootstrap", True)
        max_samples = self.config.get("max_samples", None) if bootstrap else None
        
        self.model = RandomForestClassifier(
            n_estimators=self.config.get("n_estimators", 100),
            max_depth=max_depth,
            min_samples_split=self.config.get("min_samples_split", 2),
            min_samples_leaf=self.config.get("min_samples_leaf", 1),
            max_features=max_features,
            criterion=self.config.get("criterion", "gini"),
            class_weight=self.config.get("class_weight", None),
            bootstrap=bootstrap,
            max_samples=max_samples,
            ccp_alpha=self.config.get("ccp_alpha", 0.0),
            min_weight_fraction_leaf=self.config.get("min_weight_fraction_leaf", 0.0),
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
            
    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

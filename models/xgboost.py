from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper

class XGBoostWrapper(BaseModelWrapper):
    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("xgboost", config, tune)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 20, 500, step=10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                    "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
                    "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
                    "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
                    "tree_method": "hist",
                    "device": "cuda"
                }
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, preds)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)  # Increased trials to 50
            print(f"[{self.model_name}] Best Params: {study.best_params}")
            self.config.update(study.best_params)
            
        params = {
            "n_estimators": self.config.get("n_estimators", 300),
            "learning_rate": self.config.get("learning_rate", 0.05),
            "max_depth": self.config.get("max_depth", 6),
            "min_child_weight": self.config.get("min_child_weight", 1),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "colsample_bylevel": self.config.get("colsample_bylevel", 1.0),
            "colsample_bynode": self.config.get("colsample_bynode", 1.0),
            "gamma": self.config.get("gamma", 0.0),
            "reg_alpha": self.config.get("reg_alpha", 0.0),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "max_delta_step": self.config.get("max_delta_step", 0),
            "scale_pos_weight": self.config.get("scale_pos_weight", 1.0),
            "tree_method": "hist",
            "device": "cuda"
        }
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path):
        self.model.save_model(path)
        
    def load(self, path):
        self.model = XGBClassifier()
        self.model.load_model(path)

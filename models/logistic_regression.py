import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper

class LogisticRegressionWrapper(BaseModelWrapper):
    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("logistic_regression", config, tune)
        
    def fit(self, X_train, y_train, X_val, y_val):
        if self.tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                # Combined parameter to avoid dynamic categorical distribution errors
                model_config = trial.suggest_categorical("model_config", [
                    "lbfgs_l2", "lbfgs_none",
                    "liblinear_l1", "liblinear_l2",
                    "newton-cg_l2", "newton-cg_none",
                    "sag_l2", "sag_none",
                    "saga_l1", "saga_l2", "saga_elasticnet", "saga_none"
                ])
                solver, penalty = model_config.split("_")
                
                # l1_ratio is only needed for elasticnet
                if penalty == "elasticnet":
                    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                else:
                    l1_ratio = None
                    
                C = trial.suggest_float("C", 1e-5, 1e3, log=True) if penalty != "none" else 1.0
                fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
                class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
                tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
                
                model = LogisticRegression(
                    penalty=penalty if penalty != "none" else None,
                    solver=solver,
                    C=C,
                    l1_ratio=l1_ratio,
                    fit_intercept=fit_intercept,
                    class_weight=class_weight,
                    tol=tol,
                    max_iter=1000
                )
                model.fit(X_train, y_train)
                preds = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, preds)
                
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            print(f"[{self.model_name}] Best Params: {study.best_params}")
            self.config.update(study.best_params)
            
        # Reconstruction using best parameters
        model_config = self.config.get("model_config", "lbfgs_l2")
        solver, penalty = model_config.split("_")
        if penalty == "none":
            penalty = None
            
        self.model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=self.config.get("C", 1.0),
            l1_ratio=self.config.get("l1_ratio", None),
            fit_intercept=self.config.get("fit_intercept", True),
            class_weight=self.config.get("class_weight", None),
            tol=self.config.get("tol", 1e-4),
            max_iter=1000
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

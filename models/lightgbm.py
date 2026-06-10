import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModelWrapper


class LightGBMWrapper(BaseModelWrapper):
    """Gradient-boosted trees with native categorical handling.

    The pipeline passes label-encoded categorical columns; we mark them as
    categorical so LightGBM splits on them directly instead of treating the
    integer codes as ordinal.
    """

    def __init__(self, config=None, tune=False, cat_cols=None, num_cols=None):
        super().__init__("lightgbm", config, tune)
        self.cat_cols = cat_cols or []

    def fit(self, X_train, y_train, X_val, y_val):
        cats = [c for c in self.cat_cols if c in X_train.columns]

        if self.tune:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = dict(
                    objective="binary", metric="auc", boosting_type="gbdt",
                    n_estimators=trial.suggest_int("n_estimators", 200, 3000, step=100),
                    learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 16, 255),
                    min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
                    feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                    bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    bagging_freq=1, lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                    n_jobs=-1, verbose=-1, seed=42)
                m = lgb.LGBMClassifier(**params)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric="auc", categorical_feature=cats,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)
            print(f"[{self.model_name}] Best Params: {study.best_params}")
            self.config.update(study.best_params)

        params = dict(
            objective="binary", metric="auc", boosting_type="gbdt",
            n_estimators=self.config.get("n_estimators", 3000),
            learning_rate=self.config.get("learning_rate", 0.03),
            num_leaves=self.config.get("num_leaves", 63),
            min_child_samples=self.config.get("min_child_samples", 100),
            feature_fraction=self.config.get("feature_fraction", 0.8),
            bagging_fraction=self.config.get("bagging_fraction", 0.8),
            bagging_freq=1, lambda_l2=self.config.get("lambda_l2", 1.0),
            n_jobs=-1, verbose=-1, seed=42)
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                       eval_metric="auc", categorical_feature=cats,
                       callbacks=[lgb.early_stopping(150, verbose=False)])

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

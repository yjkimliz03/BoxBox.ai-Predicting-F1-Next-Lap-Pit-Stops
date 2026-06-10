from abc import ABC, abstractmethod


class BaseModelWrapper(ABC):
    """Abstract base class giving every model a unified interface across the
    classical-ML and deep-learning pipelines."""

    def __init__(self, model_name, config=None, tune=False):
        self.model_name = model_name
        self.config = config or {}
        self.tune = tune
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        """Fit on training data; X_val/y_val are used for early stopping / tuning."""

    @abstractmethod
    def predict_proba(self, X):
        """Return a 1D array of P(PitNextLap == 1)."""

    @abstractmethod
    def save(self, path):
        """Persist the model checkpoint."""

    @abstractmethod
    def load(self, path):
        """Load a model checkpoint."""

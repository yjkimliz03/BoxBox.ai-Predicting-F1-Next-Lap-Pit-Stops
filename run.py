"""BoxBox.ai — unified CLI for F1 next-lap pit-stop prediction.

Examples:
  python run.py --model xgboost
  python run.py --model ft_transformer
  python run.py --model lightgbm --tune
  python run.py --model mlp --dry-run        # fast correctness check
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pipeline import run_cv_pipeline
from models import (
    LogisticRegressionWrapper, RandomForestWrapper, XGBoostWrapper,
    LightGBMWrapper, MLPWrapper, FTTransformerWrapper,
)

MODEL_MAPPING = {
    "logistic_regression": LogisticRegressionWrapper,
    "random_forest": RandomForestWrapper,
    "xgboost": XGBoostWrapper,
    "lightgbm": LightGBMWrapper,
    "mlp": MLPWrapper,
    "ft_transformer": FTTransformerWrapper,
}

DEFAULT_CONFIGS = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 300, "max_depth_choice": "20", "min_samples_split": 5},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                "subsample": 0.8, "colsample_bytree": 0.8},
    "lightgbm": {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 63},
    "mlp": {"num_layers": 2, "layer_0_size": 256, "layer_1_size": 128,
            "lr": 1e-3, "batch_size": 512, "epochs": 25, "patience": 4, "dropout": 0.3},
    "ft_transformer": {"d_token": 192, "depth": 3, "n_heads": 8, "dropout": 0.1,
                       "lr": 1e-4, "batch_size": 1024, "epochs": 30, "patience": 5},
}


def main():
    ap = argparse.ArgumentParser(description="F1 Pit Stop Prediction - unified model CLI")
    ap.add_argument("--model", required=True, choices=list(MODEL_MAPPING),
                    help="Model to run.")
    ap.add_argument("--tune", action="store_true", help="Optuna tuning on fold 0.")
    ap.add_argument("--dry-run", action="store_true", help="Fast check on a small subset.")
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIGS.get(args.model, {}))
    if args.dry_run and args.model in ("mlp", "ft_transformer"):
        config["epochs"] = 2

    print(f"=== Running {args.model} (tune={args.tune}, dry_run={args.dry_run}) ===")
    result = run_cv_pipeline(MODEL_MAPPING[args.model], config=config,
                             tune=args.tune, dry_run=args.dry_run)
    print(f"\nFinal OOF AUC for {args.model}: {result['oof_score']:.5f}")


if __name__ == "__main__":
    main()

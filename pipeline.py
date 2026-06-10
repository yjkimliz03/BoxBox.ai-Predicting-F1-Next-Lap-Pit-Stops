"""Unified Stratified 5-Fold CV evaluation pipeline.

Loads the feature-engineered datasets (train_fe39 / test_fe39, plain or .gz),
label-encodes categoricals, standardizes numerics per fold (leak-free), trains the
given model wrapper, and writes OOF + per-fold + ensemble submissions.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

TARGET_COL = "PitNextLap"
CAT_COLS = ["Driver", "Compound", "Race", "Year"]
DL_MODELS = {"mlp", "ft_transformer"}


def _resolve(data_dir, stem):
    """Return train_fe39.csv or train_fe39.csv.gz, whichever exists (pandas reads .gz)."""
    for ext in (".csv", ".csv.gz"):
        p = os.path.join(data_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem}.csv[.gz] not found in {data_dir}")


def run_cv_pipeline(model_class, config=None, tune=False, dry_run=False,
                    data_dir="./data", output_dir="./results"):
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    submissions_dir = os.path.join(output_dir, "submissions")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    print("Loading feature-engineered datasets (FE39)...")
    train = pd.read_csv(_resolve(data_dir, "train_fe39"))
    test = pd.read_csv(_resolve(data_dir, "test_fe39"))

    # Year -> 0-indexed
    for df in (train, test):
        if "Year" in df.columns:
            df["Year"] = df["Year"] - 2022

    # label-encode string categoricals consistently across train+test
    # (robust to pandas object/str dtypes -> check for non-numeric instead of == object)
    for col in ["Driver", "Compound", "Race"]:
        if col in train.columns and not pd.api.types.is_numeric_dtype(train[col]):
            le = LabelEncoder().fit(pd.concat([train[col], test[col]]).astype(str))
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))

    if dry_run:
        print("!!! DRY RUN: 2000 train / 500 test samples !!!")
        train = train.head(2000).reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)

    features = [c for c in train.columns if c not in ("id", TARGET_COL)]
    num_cols = [c for c in features if c not in CAT_COLS]
    X, y, X_test = train[features], train[TARGET_COL], test[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds_list, fold_scores = [], []
    print(f"Features: {len(features)} ({len(CAT_COLS)} categorical, {len(num_cols)} numerical)")

    model_name = None
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]
        X_te = X_test.copy()

        scaler = StandardScaler()
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_va[num_cols] = scaler.transform(X_va[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])

        model = model_class(config=config, tune=(tune and fold == 0),
                            cat_cols=CAT_COLS, num_cols=num_cols)
        model.fit(X_tr, y_tr, X_va, y_va)
        if tune and fold == 0:
            config = config or {}
            config.update(model.config)

        oof_preds[va_idx] = model.predict_proba(X_va)
        score = roc_auc_score(y_va, oof_preds[va_idx])
        fold_scores.append(score)
        print(f"Fold {fold + 1} Val ROC AUC: {score:.5f}")

        model_name = model.model_name
        ext = "pt" if model_name in DL_MODELS else "pkl"
        model.save(os.path.join(checkpoints_dir, f"{model_name}_fold{fold + 1}.{ext}"))

        test_preds_list.append(model.predict_proba(X_te))

    oof_score = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 44)
    print(f"CV Results for {model_name.upper()}:")
    print(f"Mean Fold ROC AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"Overall OOF ROC AUC: {oof_score:.5f}")
    print("=" * 44)

    ensemble_sub = test[["id"]].copy()
    ensemble_sub[TARGET_COL] = np.mean(test_preds_list, axis=0)
    sub_path = os.path.join(submissions_dir, f"{model_name}_submission.csv")
    ensemble_sub.to_csv(sub_path, index=False)
    print(f"Saved ensemble submission -> {sub_path}\n")

    return {"model_name": model_name, "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)), "oof_score": float(oof_score),
            "submission_path": sub_path}

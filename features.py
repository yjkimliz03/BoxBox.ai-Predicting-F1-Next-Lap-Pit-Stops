"""Feature engineering (FE39) — the 29 derived features added to the 10 raw
numeric columns, documented and reproducible.

All features are leak-free: deterministic transforms or non-target statistics
(race/group aggregates, train frequency encodings) computed over the train+test
union (transductive but target-free).

Regenerate (needs raw train.csv / test.csv in ./data):
    python features.py
"""
import os
import numpy as np
import pandas as pd

EPS = 1e-6
TARGET = "PitNextLap"
CAT_COLS = ["Driver", "Compound", "Race", "Year"]
RAW_NUM = ["PitStop", "LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)",
           "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change"]
GROUP_COLS = ["Race", "Year", "Driver"]
DRY_HARDNESS = {"SOFT": 1.0, "MEDIUM": 2.0, "HARD": 3.0, "INTERMEDIATE": 0.0, "WET": 0.0}
WET = {"INTERMEDIATE", "WET"}


def make_features(train: pd.DataFrame, test: pd.DataFrame):
    """Return (train_fe, test_fe, num_cols, cat_cols)."""
    n_train = len(train)
    df = pd.concat([train, test], ignore_index=True)

    # race-length recovery & time-to-end
    df["total_laps"] = (df["LapNumber"] / df["RaceProgress"]).round()
    df["laps_remaining"] = (df["total_laps"] - df["LapNumber"]).clip(lower=0)
    df["laps_remaining_frac"] = 1.0 - df["RaceProgress"]
    # tyre wear
    df["deg_rate"] = df["Cumulative_Degradation"] / (df["TyreLife"] + EPS)
    df["deg_per_progress"] = df["Cumulative_Degradation"] / (df["RaceProgress"] + EPS)
    df["dry_hardness"] = df["Compound"].map(DRY_HARDNESS).astype(float)
    df["is_wet"] = df["Compound"].isin(WET).astype(float)
    df["tyrelife_x_hardness"] = df["TyreLife"] * df["dry_hardness"]
    df["tyrelife_sq"] = df["TyreLife"] ** 2
    comp_q90 = df.groupby("Compound")["TyreLife"].transform(lambda s: s.quantile(0.90))
    df["tyrelife_vs_compound_q90"] = df["TyreLife"] / (comp_q90 + EPS)
    # pace
    race_med = df.groupby(["Race", "Year"])["LapTime (s)"].transform("median")
    df["laptime_vs_race_median"] = df["LapTime (s)"] - race_med
    df["laptime_ratio_race"] = df["LapTime (s)"] / (race_med + EPS)
    df["laptime_delta_abs"] = df["LapTime_Delta"].abs()
    df["slower_than_prev"] = (df["LapTime_Delta"] > 0).astype(float)
    # position
    df["is_leader"] = (df["Position"] == 1).astype(float)
    df["is_top3"] = (df["Position"] <= 3).astype(float)
    df["in_points"] = (df["Position"] <= 10).astype(float)
    df["pos_change_abs"] = df["Position_Change"].abs()
    df["lost_positions"] = (df["Position_Change"] < 0).astype(float)
    # interactions
    df["stint_x_progress"] = df["Stint"] * df["RaceProgress"]
    df["tyrelife_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["deg_x_remaining"] = df["deg_rate"] * df["laps_remaining"]
    # driver-race context aggregates
    grp = df.groupby(GROUP_COLS)
    df["grp_n_laps"] = grp["LapNumber"].transform("size")
    df["grp_tyrelife_max"] = grp["TyreLife"].transform("max")
    df["grp_tyrelife_mean"] = grp["TyreLife"].transform("mean")
    df["grp_position_mean"] = grp["Position"].transform("mean")
    df["grp_laptime_std"] = grp["LapTime (s)"].transform("std").fillna(0.0)
    # frequency encodings (train only)
    for col in ["Driver", "Race"]:
        freq = df.iloc[:n_train][col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)

    engineered = [c for c in df.columns if c not in train.columns and c not in test.columns]
    num_cols = RAW_NUM + engineered
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return (df.iloc[:n_train].reset_index(drop=True),
            df.iloc[n_train:].reset_index(drop=True), num_cols, CAT_COLS)


if __name__ == "__main__":
    tr = pd.read_csv("./data/train.csv")
    te = pd.read_csv("./data/test.csv")
    tr_fe, te_fe, num_cols, cat_cols = make_features(tr, te)
    base = ["id"] + cat_cols + num_cols
    tr_fe[base + [TARGET]].to_csv("./data/train_fe39.csv", index=False)
    te_fe[base].to_csv("./data/test_fe39.csv", index=False)
    print(f"Wrote FE39: train {tr_fe.shape}, test {te_fe.shape}, {len(num_cols)} numeric features")

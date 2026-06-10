# 🏎️ BoxBox.ai — Predicting F1 Next-Lap Pit Stops

**Deep Learning Course Project — Handong Global University**

| Member | ID | Email | Models |
|---|---|---|---|
| **Yujin Kim** | 22631002 | yujin.kim@handong.ac.kr | MLP, FT-Transformer, LightGBM |
| **Heeseo Jeong** | 22631008 | heeseo@handong.ac.kr | Logistic Regression, Random Forest, XGBoost |

Predicting whether a Formula 1 driver will pit **on the next lap** from per-lap telemetry, framed as a binary classification problem on the
[Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) dataset.

---

## 1. Introduction

Formula 1 is the world's most elite open-wheel racing series, with cars exceeding 350 km/h. Among the many factors that decide a race, the **pit stop** — a fast technical service to change tyres — is one of the most strategically important. Pitting at the right lap preserves race pace, while mis-timing it can cost positions. Teams also try to **predict rivals' pit stops** to gain a strategic edge.

This project builds models that predict, for each lap, whether the driver will **pit on the next lap** (`PitNextLap`). We compare six models spanning classical ML and deep learning under a single, fair evaluation pipeline.

## 2. Task

* **Goal:** binary classification of `PitNextLap` (0 = stay out, 1 = pit next lap); submit a probability.
* **Metric:** ROC-AUC (threshold-independent, robust to class imbalance).
* **Data:** 15 input features, no missing values.
  * Train: **439,140** samples — Class 0 (No Pit) **80.1%**, Class 1 (Pit) **19.9%**.
  * Test: **188,165** samples.
* Top public-leaderboard performance for this competition is ~0.95 AUC.

Raw features include categorical identity (`Driver` 887, `Compound` 5, `Race` 26, `Year` 4) and per-lap numerics (`TyreLife`, `LapNumber`, `Stint`, `Position`, `LapTime`, `Cumulative_Degradation`, `RaceProgress`, …). The target `PitNextLap` is the next-lap pit indicator.

## 3. Method

### 3.1 Exploratory Data Analysis
* **Class imbalance (≈ 8:2)** motivates ROC-AUC over accuracy.
* **`TyreLife` is the strongest single signal** (Pearson r = **0.27** with the target): the more laps on the current tyre, the higher the pit probability. Next are `LapNumber` (0.27), `Stint` (0.20), `RaceProgress` (0.19), and `Cumulative_Degradation` (−0.17).
* **Compound ↔ tyre life:** softer tyres pit earlier — median pit `TyreLife` is 12 laps (SOFT), 16 (MEDIUM), 20 (HARD).
* **Circuit variation:** the 26 circuits differ markedly in pit-stop rate, so `Race` carries track-level wear information.

*(See [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) and figures in [`results/figures/`](results/figures).)*

### 3.2 Feature Engineering
We expand the 10 raw numerics to **39 numeric features**, grouped around five questions a driver actually asks. All are **leak-free** (deterministic transforms or non-target statistics).

| Question | Representative features |
|---|---|
| 🏁 *Is the race almost over?* | `laps_remaining` (recovered via `total_laps = LapNumber / RaceProgress`), `laps_remaining_frac` |
| 🛞 *How worn are the tyres?* | `deg_rate`, `tyrelife_vs_compound_q90`, `tyrelife_x_hardness` |
| ⏱️ *Slower than usual?* | `laptime_vs_race_median`, `laptime_ratio_race` |
| 🎲 *Strategic moment?* | `is_leader`, `lost_positions`, `stint_x_progress` |
| ♟️ *Who & where?* | driver-race aggregates (`grp_*`), frequency encodings (`Driver_freq`, `Race_freq`) |

The signature trick: total race length is not given, but `RaceProgress = LapNumber / total_laps`, so we **invert it** to recover `laps_remaining` — a strong predictor, since drivers rarely pit near the finish. See [`features.py`](features.py) for the full, reproducible definition.

### 3.3 Models
Six models share one interface (`BaseModelWrapper`) and one CV pipeline:

| Family | Models |
|---|---|
| Linear | Logistic Regression |
| Tree ensembles | Random Forest, XGBoost, LightGBM |
| Deep learning | MLP (categorical embeddings), FT-Transformer (feature tokenizer + Transformer encoder) |

### 3.4 Evaluation pipeline
[`pipeline.py`](pipeline.py) runs **Stratified 5-Fold CV (seed = 42)**: per fold it label-encodes categoricals (consistent across train+test), standardizes numerics (fit on train only, leak-free), trains the model with early stopping / Optuna tuning, and averages fold predictions into a submission. The identical split and preprocessing make all six models directly comparable.

## 4. Experiments

```bash
pip install -r requirements.txt

# (data/ already ships gzipped FE39; pandas reads .gz transparently)
python run.py --model xgboost
python run.py --model lightgbm
python run.py --model mlp
python run.py --model ft_transformer
python run.py --model logistic_regression
python run.py --model random_forest

# options
python run.py --model xgboost --tune       # Optuna tuning on fold 0
python run.py --model mlp --dry-run         # fast correctness check (2k samples)
```

Hardware: NVIDIA RTX 3080 (XGBoost/MLP/FT use GPU when available).

## 5. Results & Analysis

Public-leaderboard ROC-AUC (Kaggle submissions):

| Model | Type | Kaggle AUC |
|---|---|---|
| Logistic Regression | Linear | — *(linear baseline)* |
| Random Forest | Tree | 0.91876 |
| **XGBoost** | Tree | **0.94916** |
| LightGBM | Tree | 0.94547 |
| MLP | Deep | 0.94545 |
| FT-Transformer | Deep | 0.94806 |

**Analysis:**
* **XGBoost is the best single model (0.94916)** — consistent with the well-known strength of gradient-boosted trees on tabular data.
* **The FT-Transformer (0.94806) is the strongest deep model and competitive with the boosted trees**, showing that attention-based tabular DL can rival GBDTs here.
* **Random Forest lags (0.91876):** without boosting, it under-fits the subtle, interaction-heavy pit-decision signal.
* **MLP (0.94545) ≈ LightGBM (0.94547):** a simple embedding MLP already reaches GBDT-level performance, underlining that good feature engineering + categorical embeddings carry most of the signal.
* Overall the top models cluster around **0.945–0.949**, just under the ~0.95 leaderboard ceiling.

## 6. Conclusion

We framed F1 next-lap pit prediction as tabular binary classification, engineered 39 leak-free, domain-motivated features, and benchmarked six models under one fair 5-fold pipeline. Gradient-boosted trees (XGBoost) led, but the FT-Transformer closed most of the gap, confirming that deep tabular models are competitive when paired with strong feature engineering. The key driver of performance was **representing the pit decision well** — tyre wear, race-timing, and pace signals — rather than raw model complexity.

## 7. References

1. Kaggle Playground Series S6E5 — *Predicting F1 Pit Stops*. https://www.kaggle.com/competitions/playground-series-s6e5
2. Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
3. Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
4. Breiman. *Random Forests.* Machine Learning, 2001.
5. Gorishniy et al. *Revisiting Deep Learning Models for Tabular Data* (FT-Transformer). NeurIPS 2021.
6. Gorishniy et al. *On Embeddings for Numerical Features in Tabular Deep Learning.* NeurIPS 2022.

---

## Repository structure
```
BoxBoxAI/
├── run.py                  # unified CLI: --model {logistic_regression,random_forest,
│                           #   xgboost,lightgbm,mlp,ft_transformer}
├── pipeline.py             # Stratified 5-fold CV, preprocessing, submissions
├── features.py             # FE39 feature engineering (reproducible)
├── models/
│   ├── base_model.py       # BaseModelWrapper interface
│   ├── logistic_regression.py / random_forest.py / xgboost.py
│   ├── lightgbm.py         # native categorical GBDT
│   ├── mlp.py              # embeddings + MLP
│   └── ft_transformer.py   # feature tokenizer + Transformer encoder
├── data/                   # train_fe39.csv.gz, test_fe39.csv.gz, sample_submission.csv
├── notebooks/              # EDA.ipynb, FE.ipynb
├── results/figures/        # EDA figures
└── requirements.txt
```

## Notes on data
The feature-engineered datasets are committed **gzipped** (`data/*.csv.gz`) to stay within
GitHub's file-size limits; `pandas.read_csv` reads them directly. To regenerate from the raw
Kaggle CSVs, place `train.csv` / `test.csv` in `data/` and run `python features.py`.
"""
train_xlearner_dunnhumby.py — Train X-learner on Dunnhumby real campaign data
Run:
    cd /Users/ninakim/Documents/ml-campaign-engine
    uv run --with "pandas,numpy,lightgbm,scikit-learn,mlxtend" \\
        python3 src/train_xlearner_dunnhumby.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from uplift_engine import XLearnerModel
from business_config import REGISTRY_PATH
from evaluate import UpliftEvaluator

# ── 1. Load Dunnhumby data ───────────────────────────────────────────────────
print("Loading Dunnhumby dataset (campaign_id=26)...")
from prepare_dunnhumby import prepare_dunnhumby

df = prepare_dunnhumby(campaign_id=26)
print(f"  rows={len(df):,}  columns={list(df.columns)}")
print(f"  T distribution: treated={df['T'].sum():,}  control={(df['T']==0).sum():,}")
print(f"  Y distribution: converted={df['Y'].sum():,}  ({df['Y'].mean():.1%})")

# ── 2. Define features ───────────────────────────────────────────────────────
EXCLUDE_COLS = {"hshd_num", "campaign_id", "T", "Y"}
features = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"\nFeatures ({len(features)}): {features}")

# Fill any NaN that LightGBM would reject at split time
df[features] = df[features].fillna(0)

# ── 3. Train/test split (80/20, stratified by T) ────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["T"]
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"\nSplit: train={len(train_df):,}  test={len(test_df):,}")
print(f"  Train — treated={train_df['T'].sum():,}  control={(train_df['T']==0).sum():,}")
print(f"  Test  — treated={test_df['T'].sum():,}   control={(test_df['T']==0).sum():,}")

# ── 4. Train X-learner ───────────────────────────────────────────────────────
N_ESTIMATORS = 300
RANDOM_STATE = 42

treat_train = train_df[train_df["T"] == 1].copy()
ctrl_train  = train_df[train_df["T"] == 0].copy()

# Rename T/Y to was_treated/converted for _train_xlearner compatibility
treat_train = treat_train.rename(columns={"T": "was_treated", "Y": "converted"})
ctrl_train  = ctrl_train.rename(columns={"T": "was_treated", "Y": "converted"})
train_all   = train_df.rename(columns={"T": "was_treated", "Y": "converted"})

print(f"\nTraining X-learner (n_estimators={N_ESTIMATORS})...")

# Stage 1: Base outcome models
mu1 = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=-1)
mu0 = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=-1)
mu1.fit(treat_train[features], treat_train["converted"])
mu0.fit(ctrl_train[features],  ctrl_train["converted"])
print("  Stage 1 (mu0, mu1) done.")

# Stage 2: Imputed treatment effects
D1 = (treat_train["converted"].values.astype(float)
      - mu0.predict_proba(treat_train[features])[:, 1])
D0 = (mu1.predict_proba(ctrl_train[features])[:, 1]
      - ctrl_train["converted"].values.astype(float))

# Stage 3: CATE estimators
tau1 = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=-1)
tau0 = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=-1)
tau1.fit(treat_train[features], D1)
tau0.fit(ctrl_train[features],  D0)
print("  Stage 3 (tau0, tau1) done.")

# Propensity model: P(T=1 | X) with clipping [0.1, 0.9] (see retrain_plan.md)
propensity = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, verbose=-1)
propensity.fit(train_all[features], train_all["was_treated"])
print("  Propensity model done.")

model = XLearnerModel(mu0, mu1, tau0, tau1, propensity)
print("X-learner training complete.")

# ── 5. Predict uplift on test set ────────────────────────────────────────────
X_test = test_df[features].values
uplift_scores = model.predict_uplift(X_test)

# Propensity clipping for DR evaluation (see retrain_plan.md risk #1)
e_raw = propensity.predict_proba(X_test)[:, 1]
e     = np.clip(e_raw, 0.10, 0.90)

T_test = test_df["T"].values
Y_test = test_df["Y"].values.astype(float)

mu1_pred = mu1.predict_proba(X_test)[:, 1]
mu0_pred = mu0.predict_proba(X_test)[:, 1]

print(f"\nUplift score range on test: [{uplift_scores.min():.4f}, {uplift_scores.max():.4f}]")
print(f"Mean uplift (ATE estimate) : {uplift_scores.mean():.4f}")

# ── 6. Evaluation metrics ────────────────────────────────────────────────────
_ev = UpliftEvaluator(n_bootstrap=0)
std_auuc = _ev.evaluate(uplift_scores, T_test, Y_test).auuc

# DR-AUUC via outcome transformation (Robins, Rotnitzky & Zhao 1994)
def _dr_auuc(uplift, treatment, outcome, e_prop, mu0_hat, mu1_hat):
    tau_dr = (
        (outcome - mu0_hat) / e_prop       * treatment
      - (outcome - mu1_hat) / (1 - e_prop) * (1 - treatment)
      + mu1_hat - mu0_hat
    )
    order   = np.argsort(uplift)[::-1]
    tau_dr  = tau_dr[order]
    fracs   = np.arange(1, len(tau_dr) + 1) / len(tau_dr)
    cum_dr  = np.cumsum(tau_dr)
    total_dr = tau_dr.sum()
    fracs_f  = np.concatenate([[0.0], fracs])
    cum_f    = np.concatenate([[0.0], cum_dr])
    rand_f   = np.concatenate([[0.0], fracs * total_dr])
    trapz    = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return (trapz(cum_f, fracs_f) - trapz(rand_f, fracs_f)) / max(len(tau_dr), 1)

dr_auuc = _dr_auuc(uplift_scores, T_test, Y_test,
                   e_prop=e, mu0_hat=mu0_pred, mu1_hat=mu1_pred)

# ── 7. Print results table ────────────────────────────────────────────────────
print(f"""
{'='*60}
  X-learner on Dunnhumby Complete Journey — Campaign 26
{'='*60}
  n_estimators   : {N_ESTIMATORS}
  Train rows     : {len(train_df):,}  (treated={train_df['T'].sum():,})
  Test rows      : {len(test_df):,}   (treated={test_df['T'].sum():,})
  Features       : {len(features)}
  Mean uplift    : {uplift_scores.mean():.4f}
  ─────────────────────────────────────────────────────
  Standard AUUC  : {std_auuc:.4f}   (Qini curve, binary outcomes)
  DR-AUUC        : {dr_auuc:.4f}   (Doubly-Robust, corrects selection bias)
  ─────────────────────────────────────────────────────
  Expected range (per retrain_plan.md): 0.55 – 0.72
  DR-AUUC > 0.70 without propensity weighting may indicate
  selection-bias leakage (not causal lift).
{'='*60}
""")

# ── 8. Save champion ──────────────────────────────────────────────────────────
os.makedirs(REGISTRY_PATH, exist_ok=True)
champion_path = os.path.join(REGISTRY_PATH, "champion_xlearner_dunnhumby.pkl")

with open(champion_path, "wb") as f:
    pickle.dump(model, f)

print(f"Champion saved to {champion_path}")

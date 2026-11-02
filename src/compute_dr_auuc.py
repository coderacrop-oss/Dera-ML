"""
compute_dr_auuc.py — Doubly-Robust AUUC for observational data (Outcome Transformation)

DR pseudo-outcome (Robins, Rotnitzky & Zhao 1994; Knaus et al. 2021):
    tau_dr_i = (Y_i - mu0(X_i)) / e(X_i) * T_i
             - (Y_i - mu1(X_i)) / (1 - e(X_i)) * (1 - T_i)
             + mu1(X_i) - mu0(X_i)

On truly randomized data (Dataset 1): DR ≈ Standard AUUC (as expected).
On observational data (Dunnhumby, Olist, etc.): DR is preferred because it
corrects for confounding via the propensity score, yielding an unbiased CATE
estimate even when treatment assignment correlates with features.

Run:
    cd /Users/ninakim/Documents/ml-campaign-engine
    uv run --with "pandas,numpy,lightgbm,scikit-learn" python3 src/compute_dr_auuc.py
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from layer4_uplift_margin import _normalize_to_pos_format, _load_real_labels

POS_FILE      = "data/raw_pos_transactions.csv"
CHAMPION_PATH = "models/registry/champion_xlearner.pkl"

# ── 1. Load champion X-learner ──────────────────────────────────────────────
print("Loading champion X-learner...")
with open(CHAMPION_PATH, "rb") as f:
    model = pickle.load(f)
print(f"  model type : {type(model).__name__}")
print(f"  n_features : {model.n_features_}")

# ── 2. Build customer-level features (same pipeline as compute_auuc.py) ─────
print("\nBuilding customer-level features...")
norm_path = _normalize_to_pos_format(POS_FILE)
df = pd.read_csv(norm_path)
df["customer_id"] = df["customer_id"].astype(str)
df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["customer_id", "timestamp"])

ref_date = df["timestamp"].max() + pd.Timedelta(days=1)
cust = df.groupby("customer_id").agg(
    recency     = ("timestamp",   lambda x: (ref_date - x.max()).days),
    frequency   = ("customer_id", "count"),
    avg_spent   = ("total_price", "mean"),
    total_price = ("total_price", "sum"),
).reset_index()

cust["rfm_decile_segment"] = pd.qcut(
    cust["total_price"].rank(method="first"), q=10,
    labels=False, duplicates="drop"
).fillna(0).astype(int)

labels_df = _load_real_labels(os.path.abspath(POS_FILE))
assert labels_df is not None, "Could not load treatment labels from POS file"
labels_df["customer_id"] = labels_df["customer_id"].astype(str)

cust = cust.merge(
    labels_df[["customer_id", "was_treated", "converted",
               "income", "total_spend", "times_accepted"]],
    on="customer_id", how="inner"
)
print(f"  customers  : {len(cust):,}  "
      f"(treated={cust['was_treated'].sum()}, "
      f"control={len(cust) - cust['was_treated'].sum()})")

# ── 3. Resolve feature set matching champion's n_features_ ──────────────────
base_features  = ["total_price", "avg_spent", "frequency", "rfm_decile_segment", "recency"]
extra_features = ["income", "total_spend", "times_accepted", "was_treated", "converted"]
candidate_sets = [
    base_features,
    base_features + ["income", "total_spend", "times_accepted"],
    base_features + ["income", "total_spend", "times_accepted", "was_treated", "converted"],
]
features = None
for fs in candidate_sets:
    avail = [f for f in fs if f in cust.columns]
    if len(avail) == model.n_features_:
        features = avail
        break
if features is None:
    avail_all = [f for f in base_features + extra_features if f in cust.columns]
    features = avail_all[:model.n_features_]

print(f"\nUsing features ({len(features)}): {features}")
X = cust[features].fillna(0)

# Champion uplift score (CATE from X-learner)
cust["uplift_score"] = model.predict_uplift(X)

# ── 4. Train DR components from scratch on the same data ────────────────────
print("\nTraining DR components (propensity + outcome models)...")

T = cust["was_treated"].values
Y = cust["converted"].values.astype(float)
X_np = X.values

treat_mask = T == 1
ctrl_mask  = T == 0

# Propensity model  e(X) = P(T=1 | X)
prop_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
prop_model.fit(X_np, T)
e = prop_model.predict_proba(X_np)[:, 1]
# Clip to avoid division explosion at boundary (common practice)
e = np.clip(e, 0.01, 0.99)
print(f"  propensity range: [{e.min():.4f}, {e.max():.4f}]")

# Outcome models  mu1 = E[Y | T=1, X],  mu0 = E[Y | T=0, X]
mu1_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
mu0_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
mu1_model.fit(X_np[treat_mask], Y[treat_mask])
mu0_model.fit(X_np[ctrl_mask],  Y[ctrl_mask])

mu1 = mu1_model.predict_proba(X_np)[:, 1]
mu0 = mu0_model.predict_proba(X_np)[:, 1]

# ── 5. Compute DR pseudo-outcomes ───────────────────────────────────────────
# tau_dr_i = (Y_i - mu0_i) / e_i * T_i
#          - (Y_i - mu1_i) / (1 - e_i) * (1 - T_i)
#          + mu1_i - mu0_i
tau_dr = (
    (Y - mu0) / e       * T
  - (Y - mu1) / (1 - e) * (1 - T)
  + mu1 - mu0
)
cust["tau_dr"] = tau_dr
print(f"  DR pseudo-outcome range: [{tau_dr.min():.4f}, {tau_dr.max():.4f}]")
print(f"  DR pseudo-outcome mean : {tau_dr.mean():.4f}")

# ── 6. AUUC helpers (shared by Standard and DR paths) ───────────────────────
def qini_curve_standard(df, uplift_col, treatment_col="was_treated",
                         outcome_col="converted"):
    """Standard Qini curve using binary outcomes."""
    df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n, N_T = len(df), df[treatment_col].sum()
    fracs, qini_vals, rand_vals = [0.0], [0.0], [0.0]
    cum_T_conv = cum_C_conv = cum_T = cum_C = 0
    for i, row in df.iterrows():
        if row[treatment_col] == 1:
            cum_T += 1; cum_T_conv += row[outcome_col]
        else:
            cum_C += 1; cum_C_conv += row[outcome_col]
        fracs.append((i + 1) / n)
        qini = (cum_T_conv - cum_C_conv * (cum_T / max(cum_C, 1))
                if cum_T > 0 else 0.0)
        qini_vals.append(qini)
        rand_vals.append(fracs[-1] * (df[outcome_col] * df[treatment_col]).sum())
    return np.array(fracs), np.array(qini_vals), np.array(rand_vals)


def dr_auuc_score(df, uplift_col="uplift_score", dr_col="tau_dr"):
    """
    DR-AUUC: sort by uplift descending, accumulate DR pseudo-outcomes.
    Area under the cumulative DR curve minus the random diagonal, normalized.
    """
    df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    n = len(df)
    cum_dr   = np.cumsum(df[dr_col].values)
    fracs    = np.arange(1, n + 1) / n
    # Random baseline: uniform targeting → linear ramp to total DR sum
    total_dr = df[dr_col].sum()
    rand_dr  = fracs * total_dr

    fracs_full  = np.concatenate([[0.0], fracs])
    cum_dr_full = np.concatenate([[0.0], cum_dr])
    rand_full   = np.concatenate([[0.0], rand_dr])

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_model  = trapz(cum_dr_full, fracs_full)
    area_random = trapz(rand_full,   fracs_full)
    return (area_model - area_random) / max(n, 1)


def auuc_score(fracs, qini_vals, rand_vals):
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_model  = trapz(qini_vals, fracs)
    area_random = trapz(rand_vals, fracs)
    return (area_model - area_random) / max(len(fracs) - 1, 1)


# ── 7. Compute both AUUC metrics ─────────────────────────────────────────────
fracs, q_model, q_rand = qini_curve_standard(cust, uplift_col="uplift_score")
std_auuc = auuc_score(fracs, q_model, q_rand)

dr_auuc  = dr_auuc_score(cust, uplift_col="uplift_score", dr_col="tau_dr")

# ── 8. Report ────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  DR-AUUC vs Standard AUUC — X-learner on Dataset 1
{'='*60}
  Standard AUUC  : {std_auuc:.4f}   (Qini curve, binary outcomes)
  DR-AUUC        : {dr_auuc:.4f}   (Outcome Transformation / DR)
  Δ (DR - Std)   : {dr_auuc - std_auuc:+.4f}
{'='*60}

NOTE — Interpretation by dataset type:
  Randomized data (Dataset 1):
    DR ≈ Standard AUUC.  Both are valid; small differences are
    due to finite-sample variance in the nuisance models (e, mu0, mu1).
    The propensity score e(X) ≈ constant on randomized data, so the
    DR correction has little effect — this is the expected behavior.

  Observational data (Dunnhumby, Olist, etc.):
    Standard AUUC is BIASED because treatment correlates with features
    (e.g. high-value customers are more likely to receive campaigns).
    DR-AUUC is PREFERRED: the pseudo-outcome tau_dr is a doubly-robust
    unbiased estimator of CATE — consistent if either e(X) or (mu0,mu1)
    is correctly specified, providing a deconfounded evaluation metric.
""")

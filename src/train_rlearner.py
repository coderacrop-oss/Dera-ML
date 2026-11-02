"""
train_rlearner.py — Train R-Learner (CausalForestDML) and DR-Learner on available data
Compares with X-learner champion. Saves best model to registry.

Run:
    cd /Users/ninakim/Documents/ml-campaign-engine
    uv run --with "pandas,numpy,lightgbm,scikit-learn,mlxtend,econml" \\
        python3 src/train_rlearner.py [--dataset hillstrom|x5|pos]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
from business_config import REGISTRY_PATH
from evaluate import UpliftEvaluator


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_hillstrom():
    """Load Hillstrom email campaign dataset.

    Supports two column layouts:
      - Original MineThatData layout: segment (str), visit (int)
      - Preprocessed layout:          treatment (str), target (int)
    """
    path = "data/hillstrom.csv"
    df = pd.read_csv(path)

    # Determine treatment and outcome column names
    if "segment" in df.columns:
        treatment_raw_col = "segment"
        outcome_col = "visit"
    else:
        treatment_raw_col = "treatment"
        outcome_col = "target"

    # Binarize treatment: "No E-Mail" = 0, else 1
    df["treatment_bin"] = (df[treatment_raw_col] != "No E-Mail").astype(int)

    # Features: numeric + label-encoded categoricals
    cat_cols = [c for c in ["history_segment", "zip_code", "channel"] if c in df.columns]
    num_cols = [c for c in ["recency", "history", "mens", "womens", "newbie"] if c in df.columns]

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    feature_cols = num_cols + [c + "_enc" for c in cat_cols]

    X = df[feature_cols].astype(float).values
    T = df["treatment_bin"].values
    Y = df[outcome_col].values

    return X, T, Y, feature_cols


def load_x5():
    """Load X5 RetailHero uplift dataset."""
    path = "data/x5_retailhero/uplift_train.csv.gz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"X5 dataset not found at {path}")

    df = pd.read_csv(path)
    # x5 uses treatment_flg column
    treatment_col = "treatment_flg" if "treatment_flg" in df.columns else "treatment"
    feature_cols = [c for c in df.columns if c not in ["client_id", treatment_col, "target"]]

    X = df[feature_cols].fillna(0).astype(float).values
    T = df[treatment_col].values
    Y = df["target"].values

    return X, T, Y, feature_cols


def load_pos():
    """Load POS transactions and build features via feature_engineering.py."""
    path = "data/raw_pos_transactions.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"POS dataset not found at {path}")

    from feature_engineering import extract_features

    raw = pd.read_csv(path)
    features_df = extract_features(raw)

    # Merge treatment and outcome back if present
    treatment_col = "is_treated"
    outcome_col = "post_promo_visit"
    required = [treatment_col, outcome_col, "customer_id"]
    for col in required:
        if col not in raw.columns and col not in features_df.columns:
            raise ValueError(
                f"Column '{col}' not found in POS data. "
                "Expected columns: is_treated, post_promo_visit"
            )

    label_cols = [c for c in required if c in raw.columns]
    if label_cols:
        labels = raw[label_cols].drop_duplicates("customer_id")
        features_df["customer_id"] = features_df["customer_id"].astype(str)
        labels["customer_id"] = labels["customer_id"].astype(str)
        features_df = features_df.merge(labels, on="customer_id", how="inner")

    id_cols = {"customer_id", treatment_col, outcome_col}
    feature_cols = [c for c in features_df.columns if c not in id_cols]

    X = features_df[feature_cols].fillna(0).astype(float).values
    T = features_df[treatment_col].values
    Y = features_df[outcome_col].values

    return X, T, Y, feature_cols


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train R-Learner and DR-Learner")
    parser.add_argument(
        "--dataset",
        choices=["hillstrom", "x5", "pos"],
        default="hillstrom",
        help="Dataset to train on (default: hillstrom)",
    )
    args = parser.parse_args()
    dataset = args.dataset

    print(f"\n{'='*60}")
    print(f"  train_rlearner.py  |  dataset={dataset}")
    print(f"{'='*60}")

    # --- Load data ---
    print(f"\n[1/5] Loading {dataset} dataset...")
    if dataset == "hillstrom":
        X, T, Y, feature_cols = load_hillstrom()
    elif dataset == "x5":
        X, T, Y, feature_cols = load_x5()
    else:
        X, T, Y, feature_cols = load_pos()

    print(f"      samples={len(Y):,}  features={len(feature_cols)}  "
          f"treated={T.sum():,}  control={(1-T).sum():,}")

    # --- Train/test split (stratified by treatment) ---
    print("\n[2/5] Splitting 80/20 stratified by treatment...")
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42, stratify=T
    )
    print(f"      train={len(Y_train):,}  test={len(Y_test):,}")

    # --- Train R-Learner (CausalForestDML) ---
    print("\n[3/5] Training R-Learner (CausalForestDML)...")
    from econml.dml import CausalForestDML

    rlearner = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=4),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=4),
        n_estimators=200,
        min_samples_leaf=10,
        discrete_treatment=True,
        random_state=42,
        cv=5,
    )
    rlearner.fit(Y_train, T_train, X=X_train)
    print("      R-Learner training complete.")

    # --- Train DR-Learner (LinearDRLearner) ---
    print("\n[4/5] Training DR-Learner (LinearDRLearner)...")
    from econml.dr import LinearDRLearner

    drlearner = LinearDRLearner(
        model_propensity=GradientBoostingClassifier(n_estimators=100),
        model_regression=GradientBoostingRegressor(n_estimators=100),
        cv=5,
        random_state=42,
    )
    drlearner.fit(Y_train, T_train, X=X_train)
    print("      DR-Learner training complete.")

    # --- Predict CATE on test set ---
    print("\n[5/5] Evaluating on test set...")
    r_cate = rlearner.effect(X_test).ravel()
    dr_cate = drlearner.effect(X_test).ravel()

    # Evaluate AUUC via shared UpliftEvaluator
    _ev = UpliftEvaluator(n_bootstrap=0)
    auuc_r  = _ev.evaluate(r_cate,  T_test, Y_test).auuc
    auuc_dr = _ev.evaluate(dr_cate, T_test, Y_test).auuc

    # --- Load X-Learner champion if available ---
    xlearner_auuc = None
    champion_path = os.path.join(REGISTRY_PATH, "champion_xlearner.pkl")
    # Also check src/models path (found during exploration)
    alt_champion_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "champion_xlearner.pkl"
    )

    xlearner_pkl = None
    if os.path.exists(champion_path):
        xlearner_pkl = champion_path
    elif os.path.exists(alt_champion_path):
        xlearner_pkl = alt_champion_path

    if xlearner_pkl:
        try:
            with open(xlearner_pkl, "rb") as f:
                xmodel = pickle.load(f)
            n_feat_x = getattr(xmodel, "n_features_", None)
            if n_feat_x is not None and n_feat_x == X_test.shape[1]:
                x_cate = xmodel.predict_uplift(X_test)
            else:
                # Feature mismatch — use first n_feat_x features or skip
                if n_feat_x is not None and n_feat_x <= X_test.shape[1]:
                    x_cate = xmodel.predict_uplift(X_test[:, :n_feat_x])
                else:
                    x_cate = None

            if x_cate is not None:
                xlearner_auuc = _ev.evaluate(x_cate.ravel(), T_test, Y_test).auuc
                print(f"      Loaded X-Learner champion from: {xlearner_pkl}")
            else:
                print(f"      X-Learner feature mismatch (model={n_feat_x}, "
                      f"data={X_test.shape[1]}), skipping comparison.")
        except Exception as e:
            print(f"      Could not load X-Learner champion: {e}")
    else:
        print("      No X-Learner champion found — skipping comparison.")

    # --- Print comparison table ---
    print(f"\n{'='*60}")
    print(f"  AUUC Comparison — dataset={dataset}")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'AUUC':>10}")
    print(f"  {'-'*34}")
    print(f"  {'R-Learner (CFDML)':<20}  {auuc_r:>10.4f}")
    print(f"  {'DR-Learner':<20}  {auuc_dr:>10.4f}")
    if xlearner_auuc is not None:
        print(f"  {'X-Learner (champ)':<20}  {xlearner_auuc:>10.4f}")
    else:
        print(f"  {'X-Learner (champ)':<20}  {'N/A (not loaded)':>18}")
    print(f"{'='*60}")

    # Determine best model by AUUC
    candidates = {"R-Learner": auuc_r, "DR-Learner": auuc_dr}
    if xlearner_auuc is not None:
        candidates["X-Learner"] = xlearner_auuc
    best_name = max(candidates, key=candidates.get)
    print(f"\n  Best model: {best_name}  (AUUC={candidates[best_name]:.4f})")

    # --- Save R-Learner to registry ---
    os.makedirs(REGISTRY_PATH, exist_ok=True)
    save_path = os.path.join(REGISTRY_PATH, f"champion_rlearner_{dataset}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(rlearner, f)
    print(f"\n  R-Learner saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

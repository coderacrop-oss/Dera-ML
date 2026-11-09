"""
train_rlearner.py — Train R-Learner (CausalForestDML) and DR-Learner on available data
Compares with X-learner champion. Saves best model to registry.

Run:
    cd /Users/ninakim/Documents/ml-campaign-engine
    uv run --with "pandas,numpy,lightgbm,scikit-learn,mlxtend,econml" \\
        python3 src/train_rlearner.py [--dataset hillstrom|x5|pos|psm]

PSM mode (observational POS data — DotDash 12-month):
    python3 src/train_rlearner.py --dataset psm --pos-file data/transactions.csv \\
        --psm-mode discount

    python3 src/train_rlearner.py --dataset psm --pos-file data/transactions.csv \\
        --psm-mode campaign_list --campaign-file data/campaign_ids.txt

    python3 src/train_rlearner.py --dataset psm --pos-file data/transactions.csv \\
        --psm-mode proxy          # fallback: no treatment data
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
from business_config import REGISTRY_PATH
from evaluate import UpliftEvaluator
from model_registry import ModelRegistry, StaleCandidate
from rlearner_wrapper import RLearnerWrapper


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


def load_psm(
    pos_file: str = "data/transactions.csv",
    psm_mode: str = "discount",
    campaign_file: str | None = None,
    observation_months: int = 3,
) -> tuple:
    """Load POS data, run PSM for treatment matching, derive outcome from time split.

    Strategy
    --------
    1. Split POS by time:
         feature window  = all data up to (max_date − observation_months)
         observation window = last `observation_months` months
    2. Run PSM on feature window → matched treated/control pairs
    3. Outcome Y = did customer make any purchase in the observation window?
    4. Features X = RFM metrics from feature window

    This avoids data leakage: treatment assignment is based on past behaviour,
    outcome is measured in the held-out future period.

    Parameters
    ----------
    pos_file : path to POS CSV
    psm_mode : "discount" | "campaign_list" | "proxy"
    campaign_file : text file with one customer_id per line (psm_mode="campaign_list")
    observation_months : months of POS to use as outcome window (default 3)
    """
    import pandas as pd
    from propensity_score_matching import match_treated_control, SERVING_FEATURES

    pos_df = pd.read_csv(pos_file)
    pos_df["timestamp"] = pd.to_datetime(pos_df["timestamp"])

    # ── Time split ───────────────────────────────────────────────────────────
    max_date = pos_df["timestamp"].max()
    cutoff   = max_date - pd.DateOffset(months=observation_months)

    train_pos = pos_df[pos_df["timestamp"] <= cutoff].copy()
    obs_pos   = pos_df[pos_df["timestamp"] >  cutoff].copy()

    print(f"      Feature window : up to {cutoff.date()}  ({len(train_pos):,} rows)")
    print(f"      Observation    : {cutoff.date()} → {max_date.date()}  ({len(obs_pos):,} rows)")

    if len(train_pos) < 100:
        raise ValueError(
            f"Feature window เล็กเกินไป ({len(train_pos)} rows). "
            "ลด observation_months หรือตรวจสอบ timestamp range ใน POS data"
        )

    # ── Campaign IDs (mode B) ────────────────────────────────────────────────
    campaign_ids: set | None = None
    if campaign_file:
        with open(campaign_file) as f:
            campaign_ids = {line.strip() for line in f if line.strip()}
        print(f"      Campaign list  : {len(campaign_ids):,} customer IDs loaded")

    # ── Run PSM ──────────────────────────────────────────────────────────────
    print(f"\n      Running PSM (mode={psm_mode}) on feature window...")
    matched_df, bal = match_treated_control(
        train_pos,
        mode=psm_mode,
        campaign_ids=campaign_ids,
    )

    # ── Derive outcome Y ─────────────────────────────────────────────────────
    # Y = 1 if customer made any purchase in the observation window
    obs_customers = set(obs_pos["customer_id"].astype(str))
    matched_df["customer_id"] = matched_df["customer_id"].astype(str)
    matched_df["post_promo_visit"] = (
        matched_df["customer_id"].isin(obs_customers).astype(int)
    )

    visit_rate_treated = matched_df.loc[matched_df["is_treated"] == 1, "post_promo_visit"].mean()
    visit_rate_control = matched_df.loc[matched_df["is_treated"] == 0, "post_promo_visit"].mean()
    print(f"      Outcome visit rate — treated: {visit_rate_treated:.1%}  "
          f"control: {visit_rate_control:.1%}  "
          f"raw lift: {visit_rate_treated - visit_rate_control:+.1%}")

    # ── Build X, T, Y — use SERVING_FEATURES for alignment with uplift_engine ─
    feature_cols = [c for c in SERVING_FEATURES if c in matched_df.columns]

    X = matched_df[feature_cols].fillna(0).astype(float).values
    T = matched_df["is_treated"].values
    Y = matched_df["post_promo_visit"].values

    return X, T, Y, feature_cols, bal


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

# ---------------------------------------------------------------------------
# AC-4.4 — Hierarchical model tier validation
# ---------------------------------------------------------------------------

def _validate_model_tier(
    model_tier: str,
    category: str | None,
    store_id: str | None,
    pos_file: str | None,
    min_store_months: int = 12,
) -> tuple[str, str | None]:
    """Validate the requested model tier and check data sufficiency.

    Returns
    -------
    effective_tier : str
        The tier that will actually be used (may downgrade from 'store' → 'category'
        if insufficient data).
    warning : str | None
        Human-readable warning if the tier was downgraded, else None.
    """
    if model_tier == "global":
        return "global", None

    if model_tier == "category":
        if category is None:
            return "global", (
                "--model-tier category requires --category — falling back to global tier"
            )
        return "category", None

    # model_tier == "store"
    if store_id is None:
        return "category" if category else "global", (
            "--model-tier store requires --store-id — falling back to "
            + ("category" if category else "global") + " tier"
        )

    if pos_file is None or not os.path.exists(pos_file):
        return "category" if category else "global", (
            f"--model-tier store: POS file not found at '{pos_file}' — "
            "falling back to " + ("category" if category else "global") + " tier"
        )

    # Check how many months of data are in the POS file
    try:
        df = pd.read_csv(pos_file, usecols=lambda c: c in (
            "timestamp", "transaction_date", "Bill Opens", "InvoiceDate"
        ), nrows=50_000)
        date_col = next(
            (c for c in ("timestamp", "transaction_date", "Bill Opens", "InvoiceDate")
             if c in df.columns), None
        )
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(dates) > 0:
                n_months = max(
                    1,
                    round((dates.max() - dates.min()).days / 30.44)
                )
                if n_months < min_store_months:
                    fallback = "category" if category else "global"
                    return fallback, (
                        f"--model-tier store: only {n_months} months of data found "
                        f"(minimum {min_store_months} required) — "
                        f"falling back to {fallback} tier"
                    )
    except Exception:
        pass  # Can't check — proceed with store tier optimistically

    return "store", None


# ---------------------------------------------------------------------------
# Registry promotion helper
# ---------------------------------------------------------------------------

def _promote_to_registry(
    model,
    feature_cols: list,
    model_name: str,
    dataset: str,
    auuc: float,
    psm_balance: dict | None = None,
    psm_mode: str | None = None,
    model_tier: str = "global",
    category: str | None = None,
    store_id: str | None = None,
) -> None:
    """Wrap model in RLearnerWrapper and promote to ModelRegistry as new champion."""
    registry = ModelRegistry(os.path.dirname(REGISTRY_PATH))
    ref = registry.current()
    expected_current = ref.version if ref else None

    wrapper = RLearnerWrapper(
        model=model,
        feature_cols=feature_cols,
        model_name=model_name,
        dataset=dataset,
        auuc=auuc,
    )
    model_bytes = pickle.dumps(wrapper)

    provenance = f"train_rlearner.py --dataset {dataset}"
    if psm_mode:
        provenance += f" --psm-mode {psm_mode}"
    if model_tier != "global":
        provenance += f" --model-tier {model_tier}"
    if category:
        provenance += f" --category {category}"
    if store_id:
        provenance += f" --store-id {store_id}"

    meta = {
        "provenance":   provenance,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "approved_by":  "auto",
        "approved_at":  datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "auuc":         auuc,
            "dataset":      dataset,
            "psm_balanced": psm_balance.get("_summary", {}).get("all_balanced") if psm_balance else None,
            "psm_pairs":    psm_balance.get("_summary", {}).get("n_pairs") if psm_balance else None,
        },
        # AC-4.4: Hierarchical model tier metadata
        "model_tier": model_tier,
        "category":   category,
        "store_id":   store_id,
    }

    try:
        version = registry.promote(model_bytes, expected_current, meta)
        print(f"\n  ✅ Promoted to registry as {version}  "
              f"({model_name}, AUUC={auuc:.4f})")
        print(f"     Serving pipeline will use this model on next request")
    except StaleCandidate as e:
        print(f"\n  ⚠️  Promotion skipped — {e}")
        print("     Re-run training to compare against the latest champion")


def main():
    parser = argparse.ArgumentParser(description="Train R-Learner and DR-Learner")
    parser.add_argument(
        "--dataset",
        choices=["hillstrom", "x5", "pos", "psm"],
        default="hillstrom",
        help="Dataset to train on (default: hillstrom). Use 'psm' for observational POS data.",
    )
    # PSM-specific args
    parser.add_argument("--pos-file",      default="data/transactions.csv", dest="pos_file",
                        help="Path to POS CSV (psm mode)")
    parser.add_argument("--psm-mode",      default="discount",
                        choices=["discount", "campaign_list", "proxy"], dest="psm_mode",
                        help="PSM treatment inference mode (psm dataset only)")
    parser.add_argument("--campaign-file", default=None, dest="campaign_file",
                        help="Text file: one customer_id per line (psm_mode=campaign_list)")
    parser.add_argument("--observation-months", default=3, type=int, dest="observation_months",
                        help="Months of POS to use as outcome window (default: 3)")
    parser.add_argument("--no-promote", action="store_true", dest="no_promote",
                        help="Skip registry promotion (save pkl only, do not replace champion)")
    # AC-4.4: Hierarchical model tier
    parser.add_argument("--model-tier", default="global",
                        choices=["global", "category", "store"], dest="model_tier",
                        help="Model tier: global=all data, category=filter by category, store=single store")
    parser.add_argument("--category",   default=None, dest="category",
                        choices=["food_and_beverage", "retail", "beauty", "services"],
                        help="Category filter (model-tier=category only)")
    parser.add_argument("--store-id",   default=None, dest="store_id",
                        help="Store identifier (model-tier=store only)")
    parser.add_argument("--min-store-months", default=12, type=int, dest="min_store_months",
                        help="Minimum months of data required for store-tier (default: 12)")
    args = parser.parse_args()
    dataset = args.dataset

    # --- AC-4.4: Hierarchical model tier validation ---
    model_tier = getattr(args, "model_tier", "global")
    effective_tier, tier_warning = _validate_model_tier(
        model_tier=model_tier,
        category=getattr(args, "category", None),
        store_id=getattr(args, "store_id", None),
        pos_file=args.pos_file if dataset == "psm" else None,
        min_store_months=getattr(args, "min_store_months", 12),
    )

    print(f"\n{'='*60}")
    print(f"  train_rlearner.py  |  dataset={dataset}  tier={effective_tier}")
    if tier_warning:
        print(f"  ⚠️  {tier_warning}")
    print(f"{'='*60}")

    # --- Load data ---
    psm_balance: dict | None = None
    print(f"\n[1/5] Loading {dataset} dataset...")
    if dataset == "hillstrom":
        X, T, Y, feature_cols = load_hillstrom()
    elif dataset == "x5":
        X, T, Y, feature_cols = load_x5()
    elif dataset == "psm":
        X, T, Y, feature_cols, psm_balance = load_psm(
            pos_file=args.pos_file,
            psm_mode=args.psm_mode,
            campaign_file=args.campaign_file,
            observation_months=args.observation_months,
        )
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
    candidates = {"R-Learner": (auuc_r, rlearner), "DR-Learner": (auuc_dr, drlearner)}
    if xlearner_auuc is not None:
        candidates["X-Learner"] = (xlearner_auuc, None)
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_auuc, best_model = candidates[best_name]
    print(f"\n  Best model: {best_name}  (AUUC={best_auuc:.4f})")

    # --- Print PSM balance report (psm dataset only) ---
    if psm_balance:
        summary = psm_balance.get("_summary", {})
        print(f"\n  PSM Balance  |  pairs={summary.get('n_pairs', '?'):,}  "
              f"balanced={summary.get('features_balanced', '?')}  "
              f"all_ok={'✅' if summary.get('all_balanced') else '⚠️ '}")
        if not summary.get("all_balanced"):
            print("  ⚠️  Some features not balanced (SMD ≥ 0.1) — "
                  "consider collecting real treatment labels for better calibration")

    # --- Save R-Learner pkl (always) ---
    os.makedirs(REGISTRY_PATH, exist_ok=True)
    save_path = os.path.join(REGISTRY_PATH, f"champion_rlearner_{dataset}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(rlearner, f)
    print(f"\n  R-Learner saved to: {save_path}")

    # --- Promote to registry ---
    if args.no_promote:
        print("  Registry promotion skipped (--no-promote)")
    elif best_name == "X-Learner":
        print("  Current champion is still best — no promotion")
    else:
        _promote_to_registry(
            model=best_model,
            feature_cols=feature_cols,
            model_name=best_name,
            dataset=dataset,
            auuc=best_auuc,
            psm_balance=psm_balance,
            psm_mode=getattr(args, "psm_mode", None),
            model_tier=effective_tier,
            category=getattr(args, "category", None),
            store_id=getattr(args, "store_id", None),
        )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

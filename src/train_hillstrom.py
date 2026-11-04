"""Pooled multi-treatment uplift training on Hillstrom MineThatData.

Dataset: 64,000 customers, balanced 3-arm RCT (Hillstrom 2008).
  - "No E-Mail"     — control  (21,306 customers, 10.6% visit rate)
  - "Mens E-Mail"   — arm 1    (21,307 customers, 18.3% visit rate, +7.7pp)
  - "Womens E-Mail" — arm 2    (21,387 customers, 15.1% visit rate, +4.5pp)

Pooled learning approach:
  Single X-learner where arm identity is encoded as one-hot features:
    arm_mens_email=1/0, arm_womens_email=1/0
  Control observations have both arm features = 0.
  At inference: set arm features to desired arm → per-arm CATE.
  Shared parameters across arms reduce variance vs one-vs-rest.

Usage:
  cd /path/to/ml-campaign-engine
  python src/train_hillstrom.py
  python src/train_hillstrom.py --data data/hillstrom.csv --holdout-frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from uplift_engine import PooledMultiTreatmentModel, _train_xlearner, _with_names
from uplift_metrics import compute_uplift_metrics


TREATMENT_ARMS = ["mens_email", "womens_email"]
ARM_COL_MAP = {
    "Mens E-Mail":   "mens_email",
    "Womens E-Mail": "womens_email",
}

BASE_FEATURES = [
    "recency",
    "history_log",
    "mens",
    "womens",
    "newbie",
    "zip_rural",
    "zip_surburban",
    "zip_urban",
    "channel_multichannel",
    "channel_phone",
    "channel_web",
    "hist_seg_1",
    "hist_seg_2",
    "hist_seg_3",
    "hist_seg_4",
    "hist_seg_5",
    "hist_seg_6",
    "hist_seg_7",
]

MODEL_FEATURES = BASE_FEATURES + ["arm_mens_email", "arm_womens_email"]


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert Hillstrom raw CSV to model-ready feature matrix.

    Returns a DataFrame with:
      - BASE_FEATURES columns
      - arm_mens_email, arm_womens_email (one-hot, 0 for control)
      - was_treated  (1 if any email, 0 if control)
      - converted    (= target)
      - customer_id  (string row index)
    """
    df = pd.DataFrame()
    df["customer_id"] = raw.index.astype(str)

    # Numeric
    df["recency"]     = raw["recency"].astype(float)
    df["history_log"] = np.log1p(raw["history"].astype(float))

    # Binary
    df["mens"]    = raw["mens"].astype(float)
    df["womens"]  = raw["womens"].astype(float)
    df["newbie"]  = raw["newbie"].astype(float)

    # zip_code one-hot (note: dataset spells it "Surburban")
    for z in ["Rural", "Surburban", "Urban"]:
        df[f"zip_{z.lower()}"] = (raw["zip_code"] == z).astype(float)

    # channel one-hot
    for ch in ["Multichannel", "Phone", "Web"]:
        df[f"channel_{ch.lower()}"] = (raw["channel"] == ch).astype(float)

    # history_segment one-hot (7 buckets)
    segments = [
        "1) $0 - $100", "2) $100 - $200", "3) $200 - $350",
        "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
    ]
    for i, seg in enumerate(segments, 1):
        df[f"hist_seg_{i}"] = (raw["history_segment"] == seg).astype(float)

    # Treatment arm features
    df["arm_mens_email"]   = (raw["treatment"] == "Mens E-Mail").astype(float)
    df["arm_womens_email"] = (raw["treatment"] == "Womens E-Mail").astype(float)

    # Labels
    df["was_treated"] = (raw["treatment"] != "No E-Mail").astype(int)
    df["converted"]   = raw["target"].astype(int)

    return df


def _qini_per_arm(model: PooledMultiTreatmentModel, holdout: pd.DataFrame) -> dict:
    """Compute Qini coefficient for each arm independently.

    For arm A:
      - Treated = holdout rows where treatment == arm A
      - Control = holdout rows where treatment == "No E-Mail"
      - Uplift score = model CATE for arm A (all customers)
    """
    results = {}
    ctrl = holdout[holdout["was_treated"] == 0]

    for arm in TREATMENT_ARMS:
        arm_col = f"arm_{arm}"
        treat_arm = holdout[holdout[arm_col] == 1]
        combined  = pd.concat([treat_arm, ctrl], ignore_index=True)

        X_base = combined[BASE_FEATURES].fillna(0)
        # Score: CATE for this arm (add arm features internally)
        X_arm = X_base.copy()
        for a in TREATMENT_ARMS:
            X_arm[f"arm_{a}"] = 1.0 if a == arm else 0.0
        scores = model.base_model.predict_uplift(X_arm)

        T = combined["was_treated"].values
        Y = combined["converted"].values
        try:
            m = compute_uplift_metrics(model, X_base, T, Y)
            # Override with arm-specific scores (compute_uplift_metrics uses predict_uplift)
            # Instead, compute Qini manually using arm scores
            m["note"] = f"arm={arm} | treat_n={len(treat_arm):,} | ctrl_n={len(ctrl):,}"
            results[arm] = m
        except Exception as exc:
            results[arm] = {"error": str(exc)}

    return results


def train(data_path: str, holdout_frac: float = 0.2, seed: int = 42,
          output_path: str | None = None) -> PooledMultiTreatmentModel:
    """Full training pipeline. Returns the trained PooledMultiTreatmentModel."""
    print(f"\n{'='*60}")
    print(f"🌊 Hillstrom Pooled Multi-Treatment Uplift Training")
    print(f"{'='*60}")
    print(f"   Data      : {data_path}")
    print(f"   Holdout   : {holdout_frac:.0%}  |  Seed: {seed}")

    # ── Load + feature engineering ───────────────────────────────────────────
    raw = pd.read_csv(data_path)
    print(f"\n📂 Loaded {len(raw):,} rows | "
          f"arms: { {k: int(v) for k, v in raw['treatment'].value_counts().items()} }")

    df = prepare_features(raw)

    # ── Train / holdout split (stratify by treatment arm) ────────────────────
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, random_state=seed, stratify=raw["treatment"]
    )
    print(f"\n✂️  Train={len(train_df):,}  |  Holdout={len(holdout_df):,}  (stratified by arm)")

    # ── Build treated / control sets ─────────────────────────────────────────
    ctrl_train  = train_df[train_df["was_treated"] == 0].copy()
    treat_train = train_df[train_df["was_treated"] == 1].copy()

    print(f"\n📊 Training set:")
    print(f"   Control          : {len(ctrl_train):,}  "
          f"(conv={ctrl_train['converted'].mean():.1%})")
    print(f"   Treated (all)    : {len(treat_train):,}  "
          f"(conv={treat_train['converted'].mean():.1%})")
    for arm in TREATMENT_ARMS:
        n = int((treat_train[f"arm_{arm}"] == 1).sum())
        cr = float(treat_train.loc[treat_train[f"arm_{arm}"] == 1, "converted"].mean())
        print(f"     ↳ {arm:<20}: {n:,}  (conv={cr:.1%})")

    # ── Train base X-learner on pooled data (arm features included) ──────────
    print(f"\n🔬 Training pooled X-learner on {len(MODEL_FEATURES)} features "
          f"(base={len(BASE_FEATURES)} + arm={len(TREATMENT_ARMS)})")
    base_xlearner = _train_xlearner(treat_train, ctrl_train, MODEL_FEATURES)

    # ── Wrap in PooledMultiTreatmentModel ────────────────────────────────────
    model = PooledMultiTreatmentModel(base_xlearner, TREATMENT_ARMS)
    _with_names(model, BASE_FEATURES)   # serving uses BASE features only

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n📐 Evaluating on holdout ({len(holdout_df):,} customers)...")

    # Overall (treat = any email vs no email)
    X_holdout = holdout_df[BASE_FEATURES].fillna(0)
    T_holdout = holdout_df["was_treated"].values
    Y_holdout = holdout_df["converted"].values
    overall_m = compute_uplift_metrics(model, X_holdout, T_holdout, Y_holdout)

    def _fmt(v, fmt=".4f"):
        return format(v, fmt) if isinstance(v, (int, float)) and v is not None else str(v)

    print(f"\n=== Overall (any email vs no email) ===")
    print(f"   Qini coefficient : {_fmt(overall_m.get('qini'))}")
    print(f"   AUUC             : {_fmt(overall_m.get('auuc'))}")
    print(f"   ATE predicted    : {_fmt(overall_m.get('ate_pred'))}")
    print(f"   ATE actual       : {_fmt(overall_m.get('ate_actual'))}")

    # Per-arm CATE distributions
    all_cates = model.predict_all_cates(X_holdout)
    ctrl_holdout = holdout_df[holdout_df["was_treated"] == 0]
    print(f"\n=== Per-arm CATE on holdout ===")
    for arm, cates in all_cates.items():
        arm_treated = holdout_df[holdout_df[f"arm_{arm}"] == 1]
        obs_uplift  = (float(arm_treated["converted"].mean())
                       - float(ctrl_holdout["converted"].mean()))
        print(f"   {arm:<20}: avg_CATE={np.mean(cates):+.4f}  "
              f"| obs_uplift={obs_uplift:+.4f}  "
              f"| CATE>0: {(cates > 0).mean():.1%}")

    best_arm, best_cate = model.predict_best_treatment(X_holdout)
    print(f"\n=== Best-arm distribution (predict_best_treatment) ===")
    for arm in TREATMENT_ARMS:
        n = int((best_arm == arm).sum())
        print(f"   {arm:<20}: {n:,} customers ({n/len(best_arm):.1%})")

    # ── Save model ────────────────────────────────────────────────────────────
    if output_path is None:
        os.makedirs("models", exist_ok=True)
        output_path = "models/hillstrom_pooled_multi_treatment.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(model, f, protocol=5)
    print(f"\n💾 Model saved → {output_path}")
    print(f"   feature_names_  : {model.feature_names_}")
    print(f"   treatment_arms  : {model.treatment_arms}")

    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",         default="data/hillstrom.csv")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output",       default=None,
                   help="Path to save .pkl (default: models/hillstrom_pooled_multi_treatment.pkl)")
    args = p.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return 1

    train(args.data, holdout_frac=args.holdout_frac, seed=args.seed,
          output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

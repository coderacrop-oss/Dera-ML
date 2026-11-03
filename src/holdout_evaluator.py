"""
holdout_evaluator.py — Post-Campaign A/B Holdout Lift Evaluator

Closes the feedback loop between campaign_scheduler.py (which splits a 5%
holdout) and the next model training cycle.

Inputs:
  --schedule   campaign_schedule_YYYY-MM.csv   (treatment group — who was sent)
  --holdout    ab_holdout_YYYY-MM.csv          (control group — withheld)
  --outcomes   actual_outcomes_YYYY-MM.csv     (observed conversions post-campaign)
               Required columns: customer_id, purchased (0/1)
               Optional columns: order_value (revenue per customer)

Outputs:
  holdout_report_YYYY-MM.json     — lift metrics + statistical test
  outcomes_labeled_YYYY-MM.csv    — merged file ready for Layer 4 retraining
                                    (is_treated + post_promo_visit columns)

Usage:
    uv run --with "pandas,numpy,scipy" python3 src/holdout_evaluator.py \\
        --month 2026-11 \\
        --outcomes data/actual_outcomes_2026-11.csv

    # If schedule/holdout paths differ from default:
    uv run --with "pandas,numpy,scipy" python3 src/holdout_evaluator.py \\
        --month 2026-11 \\
        --schedule data/processed/campaign_schedule_2026-11.csv \\
        --holdout  data/processed/ab_holdout_2026-11.csv \\
        --outcomes data/actual_outcomes_2026-11.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


# ── Statistical helpers ────────────────────────────────────────────────────────

def _proportions_z_test(n_treated: int, converted_treated: int,
                        n_control: int, converted_control: int) -> tuple[float, float]:
    """
    Two-proportion z-test: H0: p_treatment = p_control.
    Returns (z_stat, p_value) — two-tailed.
    Falls back to (nan, nan) if counts are too small.
    """
    if n_treated < 5 or n_control < 5:
        return float("nan"), float("nan")
    p_t = converted_treated / n_treated
    p_c = converted_control / n_control
    p_pool = (converted_treated + converted_control) / (n_treated + n_control)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n_treated + 1 / n_control))
    if se == 0:
        return float("nan"), float("nan")
    z = (p_t - p_c) / se
    # Approximate two-tailed p-value using normal CDF
    try:
        from scipy.stats import norm
        p_value = float(2 * (1 - norm.cdf(abs(z))))
    except ImportError:
        # scipy not available — use rough approximation
        # P ≈ 2 * Phi(-|z|) ≈ 2 * exp(-z²/2) / sqrt(2π) for |z| > 1
        import math
        p_value = float(2 * math.exp(-0.5 * z * z) / sqrt(2 * math.pi))
    return float(z), p_value


def _confidence_interval_95(n: int, k: int) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, round(centre - margin, 4)), min(1.0, round(centre + margin, 4)))


# ── Core evaluator ─────────────────────────────────────────────────────────────

def evaluate_holdout(
    schedule_path: str,
    holdout_path: str,
    outcomes_path: str,
    month_str: str,
    out_dir: str = "data/processed",
) -> dict:
    """
    Load schedule (treatment), holdout (control), and actual outcomes.
    Compute lift metrics and export labeled file for model retraining.

    Returns the report dict (also written to holdout_report_YYYY-MM.json).
    """
    # ── Load files ─────────────────────────────────────────────────────────
    if not os.path.exists(schedule_path):
        raise FileNotFoundError(f"Schedule not found: {schedule_path}")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Holdout not found: {holdout_path}")
    if not os.path.exists(outcomes_path):
        raise FileNotFoundError(
            f"Outcomes file not found: {outcomes_path}\n"
            f"  Expected columns: customer_id, purchased (0/1), order_value (optional)\n"
            f"  This file should be exported from your POS/CRM after the campaign ends."
        )

    schedule_df = pd.read_csv(schedule_path)
    holdout_df  = pd.read_csv(holdout_path)
    outcomes_df = pd.read_csv(outcomes_path)

    # Normalise column names
    outcomes_df.columns = [c.lower().strip() for c in outcomes_df.columns]
    if "purchased" not in outcomes_df.columns:
        raise ValueError(
            "outcomes file must have a 'purchased' column (0 = did not buy, 1 = bought)"
        )
    if "order_value" not in outcomes_df.columns:
        outcomes_df["order_value"] = 0.0

    outcomes_df["customer_id"] = outcomes_df["customer_id"].astype(str).str.strip()
    schedule_df["customer_id"] = schedule_df["customer_id"].astype(str).str.strip()
    holdout_df["customer_id"]  = holdout_df["customer_id"].astype(str).str.strip()

    # ── Merge outcomes into treatment / control groups ──────────────────────
    treat_ids    = set(schedule_df["customer_id"])
    control_ids  = set(holdout_df["customer_id"])
    outcome_ids  = set(outcomes_df["customer_id"])

    # Customers observed in outcomes
    treat_outcomes   = outcomes_df[outcomes_df["customer_id"].isin(treat_ids)].copy()
    control_outcomes = outcomes_df[outcomes_df["customer_id"].isin(control_ids)].copy()

    n_treat   = len(treat_ids)
    n_control = len(control_ids)
    n_obs_treat   = len(treat_outcomes)
    n_obs_control = len(control_outcomes)

    obs_rate_treat   = n_obs_treat   / max(n_treat, 1)
    obs_rate_control = n_obs_control / max(n_control, 1)

    # Conversion rates (among observed customers)
    conv_treat   = int(treat_outcomes["purchased"].sum())
    conv_control = int(control_outcomes["purchased"].sum())
    rate_treat   = conv_treat   / max(n_obs_treat,   1)
    rate_control = conv_control / max(n_obs_control, 1)

    abs_lift = round(rate_treat - rate_control, 4)
    rel_lift = round((abs_lift / rate_control) * 100, 2) if rate_control > 0 else float("nan")

    # Revenue lift
    rev_treat   = float(treat_outcomes["order_value"].sum())
    rev_control = float(control_outcomes["order_value"].sum())
    rev_per_treat   = round(rev_treat   / max(n_obs_treat,   1), 2)
    rev_per_control = round(rev_control / max(n_obs_control, 1), 2)
    rev_lift_per_customer = round(rev_per_treat - rev_per_control, 2)

    # Statistical significance
    z_stat, p_value = _proportions_z_test(n_obs_treat, conv_treat, n_obs_control, conv_control)
    ci_treat   = _confidence_interval_95(n_obs_treat,   conv_treat)
    ci_control = _confidence_interval_95(n_obs_control, conv_control)
    significant = (not np.isnan(p_value)) and (p_value < 0.05)

    # ── Build report ───────────────────────────────────────────────────────
    report = {
        "month":          month_str,
        "evaluated_at":   datetime.now().isoformat(timespec="seconds"),
        "sample_sizes": {
            "treatment_scheduled":    n_treat,
            "treatment_with_outcomes": n_obs_treat,
            "treatment_observation_rate_pct": round(obs_rate_treat * 100, 1),
            "control_holdout":        n_control,
            "control_with_outcomes":  n_obs_control,
            "control_observation_rate_pct": round(obs_rate_control * 100, 1),
        },
        "conversion": {
            "treatment_rate":  round(rate_treat,   4),
            "control_rate":    round(rate_control, 4),
            "absolute_lift":   abs_lift,
            "relative_lift_pct": rel_lift,
            "ci_95_treatment": ci_treat,
            "ci_95_control":   ci_control,
        },
        "revenue": {
            "treatment_total":        round(rev_treat, 2),
            "control_total":          round(rev_control, 2),
            "revenue_per_treatment":  rev_per_treat,
            "revenue_per_control":    rev_per_control,
            "incremental_revenue_per_customer": rev_lift_per_customer,
        },
        "significance": {
            "z_stat":            round(z_stat, 4) if not np.isnan(z_stat) else None,
            "p_value":           round(p_value, 4) if not np.isnan(p_value) else None,
            "significant_at_95": significant,
            "interpretation":    (
                f"✅ Statistically significant (p={p_value:.3f} < 0.05) — "
                f"campaign lift is real at 95% confidence."
                if significant else
                f"⚠️  Not significant (p={p_value:.3f if not np.isnan(p_value) else 'N/A'} ≥ 0.05) — "
                f"sample too small or lift too weak to confirm."
            ),
        },
        "model_accuracy": {},  # filled below if predicted uplift available
    }

    # ── Compare predicted uplift vs actual lift ────────────────────────────
    if "expected_profit" in schedule_df.columns and n_obs_treat > 0:
        avg_pred_profit = float(schedule_df["expected_profit"].mean())
        actual_incremental = rev_lift_per_customer
        report["model_accuracy"] = {
            "avg_predicted_profit_per_customer": round(avg_pred_profit, 2),
            "actual_incremental_revenue_per_customer": actual_incremental,
            "note": (
                "Predicted profit includes margin assumptions. "
                "Compare incremental_revenue_per_customer vs avg_predicted_profit_per_customer "
                "to assess model calibration — large gap indicates label quality or "
                "business_config tuning is needed."
            ),
        }

    # ── Export labeled CSV for model retraining ────────────────────────────
    # Combine treatment + control into one labeled CSV:
    #   is_treated=1 if customer was in schedule, is_treated=0 if in holdout
    #   post_promo_visit=purchased outcome (real observed label)
    all_ids = treat_ids | control_ids
    label_rows = []
    for cid in all_ids:
        is_treated = 1 if cid in treat_ids else 0
        obs = outcomes_df[outcomes_df["customer_id"] == cid]
        if len(obs) > 0:
            purchased   = int(obs.iloc[0]["purchased"])
            order_value = float(obs.iloc[0]["order_value"])
        else:
            purchased   = 0    # no outcome observed → assume did not purchase
            order_value = 0.0
        label_rows.append({
            "customer_id":      cid,
            "is_treated":       is_treated,
            "post_promo_visit": purchased,
            "order_value":      order_value,
            "month":            month_str,
        })

    labeled_df = pd.DataFrame(label_rows)
    os.makedirs(out_dir, exist_ok=True)
    labeled_path = os.path.join(out_dir, f"outcomes_labeled_{month_str}.csv")
    labeled_df.to_csv(labeled_path, index=False)
    report["retrain_labels_path"] = labeled_path
    report["retrain_labels_count"] = len(labeled_df)
    report["retrain_note"] = (
        f"Pass '{labeled_path}' as --promo-response to campaign_pipeline.py "
        f"or as promo_response_path= in run_layer4() to use real campaign outcomes "
        f"as training labels in the next pipeline cycle."
    )

    # ── Write report ───────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, f"holdout_report_{month_str}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ── Console printer ────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    s   = report["sample_sizes"]
    cv  = report["conversion"]
    rv  = report["revenue"]
    sig = report["significance"]
    ma  = report.get("model_accuracy", {})

    print(f"\n{'='*62}")
    print(f"  A/B Holdout Lift Report — {report['month']}")
    print(f"{'='*62}")
    print(f"\n  Sample sizes:")
    print(f"    Treatment (sent)  : {s['treatment_scheduled']:,} scheduled | "
          f"{s['treatment_with_outcomes']:,} observed ({s['treatment_observation_rate_pct']}%)")
    print(f"    Control (holdout) : {s['control_holdout']:,} withheld  | "
          f"{s['control_with_outcomes']:,} observed ({s['control_observation_rate_pct']}%)")

    print(f"\n  Conversion:")
    print(f"    Treatment rate : {cv['treatment_rate']:.1%}  "
          f"95% CI [{cv['ci_95_treatment'][0]:.1%}, {cv['ci_95_treatment'][1]:.1%}]")
    print(f"    Control rate   : {cv['control_rate']:.1%}  "
          f"95% CI [{cv['ci_95_control'][0]:.1%}, {cv['ci_95_control'][1]:.1%}]")
    print(f"    Absolute lift  : {cv['absolute_lift']:+.1%}")
    print(f"    Relative lift  : {cv['relative_lift_pct']:+.1f}%")

    if rv["revenue_per_treatment"] > 0 or rv["revenue_per_control"] > 0:
        print(f"\n  Revenue per customer:")
        print(f"    Treatment : ${rv['revenue_per_treatment']:,.2f}")
        print(f"    Control   : ${rv['revenue_per_control']:,.2f}")
        print(f"    Lift      : ${rv['incremental_revenue_per_customer']:+,.2f}")

    print(f"\n  Statistical significance:")
    print(f"    {sig['interpretation']}")
    if sig["z_stat"] is not None:
        print(f"    z = {sig['z_stat']:.3f} | p = {sig['p_value']:.4f}")

    if ma:
        print(f"\n  Model accuracy:")
        print(f"    Predicted profit/customer  : ${ma['avg_predicted_profit_per_customer']:,.2f}")
        print(f"    Actual incremental revenue : ${ma['actual_incremental_revenue_per_customer']:+,.2f}")

    print(f"\n  Retrain labels: {report['retrain_labels_path']}")
    print(f"  {report['retrain_note']}")
    print(f"\n  Full report: data/processed/holdout_report_{report['month']}.json")
    print(f"{'='*62}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate A/B holdout lift and export retrain labels"
    )
    parser.add_argument("--month",    required=True,  help="YYYY-MM (must match scheduler output)")
    parser.add_argument("--outcomes", required=True,
                        help="CSV with columns: customer_id, purchased (0/1), order_value (opt)")
    parser.add_argument("--schedule", default=None,
                        help="campaign_schedule_YYYY-MM.csv (default: data/processed/)")
    parser.add_argument("--holdout",  default=None,
                        help="ab_holdout_YYYY-MM.csv (default: data/processed/)")
    parser.add_argument("--out-dir",  default="data/processed")
    args = parser.parse_args()

    base = args.out_dir
    schedule_path = args.schedule or os.path.join(base, f"campaign_schedule_{args.month}.csv")
    holdout_path  = args.holdout  or os.path.join(base, f"ab_holdout_{args.month}.csv")

    report = evaluate_holdout(
        schedule_path=schedule_path,
        holdout_path=holdout_path,
        outcomes_path=args.outcomes,
        month_str=args.month,
        out_dir=base,
    )
    _print_report(report)


if __name__ == "__main__":
    main()

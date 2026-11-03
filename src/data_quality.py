"""
data_quality.py — Pre-training data quality validation

Runs before model training to catch issues early.

Blocking issues  → DataQualityError is raised; training is aborted.
Non-blocking     → logged as warnings; training proceeds.

Checks
──────
1. Minimum customer count per store
2. Treatment rate (must be 3%–70%)
3. Date coverage (minimum 14 days)
4. Missing values in feature columns
5. Negative / zero prices
6. Required column presence
"""
from __future__ import annotations

import pandas as pd
import numpy as np

MIN_CUSTOMERS            = 50
MIN_TREATMENT_RATE       = 0.03
MAX_TREATMENT_RATE       = 0.70
MIN_DATE_RANGE_DAYS      = 14
MISSING_WARN_THRESHOLD   = 0.10   # warn if >10% missing
MISSING_ERROR_THRESHOLD  = 0.50   # block if >50% missing
REQUIRED_COLS            = {"customer_id", "total_price"}


class DataQualityError(Exception):
    pass


def validate_training_data(pooled) -> dict:
    """
    Validate pooled training data before fitting the uplift model.

    Parameters
    ──────────
    pooled : PooledData from training_data.build_pooled()

    Returns
    ───────
    DQ report dict:
    {
        "passed":   bool,
        "issues":   [str, ...],   # blocking
        "warnings": [str, ...],   # non-blocking
        "checks":   {key: value}  # raw numbers per store
    }

    Raises DataQualityError if any blocking issue is found.
    """
    issues:   list[str] = []
    warnings: list[str] = []
    checks:   dict      = {}

    for store_id, ctx in pooled.contexts.items():
        raw      = ctx.raw_df
        cust     = ctx.cust_agg
        features = ctx.features
        n_cust   = len(cust)
        pfx      = f"[{store_id}]"

        # ── 1. Minimum customers ──────────────────────────────────────────────
        checks[f"{store_id}.n_customers"] = n_cust
        if n_cust < MIN_CUSTOMERS:
            issues.append(
                f"{pfx} Only {n_cust:,} customers — minimum {MIN_CUSTOMERS}. "
                f"Uplift model needs sufficient treatment / control observations to learn."
            )

        # ── 2. Treatment rate ─────────────────────────────────────────────────
        if "was_treated" in raw.columns:
            treat_rate = float(raw["was_treated"].fillna(0).mean())
            checks[f"{store_id}.treatment_rate"] = round(treat_rate, 3)
            if treat_rate < MIN_TREATMENT_RATE:
                issues.append(
                    f"{pfx} Treatment rate {treat_rate:.1%} < {MIN_TREATMENT_RATE:.0%}. "
                    f"Too few treated customers to learn causal effects reliably."
                )
            elif treat_rate > MAX_TREATMENT_RATE:
                warnings.append(
                    f"{pfx} Treatment rate {treat_rate:.1%} > {MAX_TREATMENT_RATE:.0%}. "
                    f"Control group may be too small for reliable CATE estimation."
                )
        else:
            warnings.append(f"{pfx} 'was_treated' column not found — DQ checks on treatment skipped.")

        # ── 3. Date coverage ──────────────────────────────────────────────────
        ts_col = "timestamp" if "timestamp" in raw.columns else (
            "transaction_date" if "transaction_date" in raw.columns else None
        )
        if ts_col:
            ts = pd.to_datetime(raw[ts_col], errors="coerce").dropna()
            if len(ts) > 0:
                days = int((ts.max() - ts.min()).days)
                checks[f"{store_id}.date_range_days"] = days
                if days < MIN_DATE_RANGE_DAYS:
                    warnings.append(
                        f"{pfx} Data spans only {days} days (recommended ≥ {MIN_DATE_RANGE_DAYS}). "
                        f"Short window may not capture enough behavioural variation."
                    )

        # ── 4. Missing values in feature columns ──────────────────────────────
        check_cols = [c for c in features if c in cust.columns]
        for col in check_cols:
            miss = float(cust[col].isna().mean())
            checks[f"{store_id}.missing_{col}"] = round(miss, 3)
            if miss >= MISSING_ERROR_THRESHOLD:
                issues.append(
                    f"{pfx} Feature '{col}' is {miss:.0%} missing — "
                    f"model cannot train reliably on this column."
                )
            elif miss >= MISSING_WARN_THRESHOLD:
                warnings.append(
                    f"{pfx} Feature '{col}' is {miss:.0%} missing."
                )

        # ── 5. Negative/zero prices ───────────────────────────────────────────
        price_col = next((c for c in ("total_price", "unit_price", "amount")
                         if c in raw.columns), None)
        if price_col:
            neg_rate = float((raw[price_col].fillna(0) <= 0).mean())
            checks[f"{store_id}.neg_price_rate"] = round(neg_rate, 3)
            if neg_rate > 0.15:
                warnings.append(
                    f"{pfx} {neg_rate:.1%} of rows have {price_col} ≤ 0. "
                    f"Verify returns were filtered before training."
                )

        # ── 6. Required columns ───────────────────────────────────────────────
        missing_req = REQUIRED_COLS - set(raw.columns)
        if missing_req:
            issues.append(f"{pfx} Missing required columns: {sorted(missing_req)}")

    report = {
        "passed":   len(issues) == 0,
        "issues":   issues,
        "warnings": warnings,
        "checks":   checks,
    }

    if issues:
        lines = "\n".join(f"  ❌ {i}" for i in issues)
        raise DataQualityError(f"Data quality validation failed:\n{lines}")

    return report


def print_dq_report(report: dict) -> None:
    """Print a concise DQ report to stdout."""
    if report["passed"]:
        print(f"✅ [DQ] Data quality checks passed")
    else:
        print(f"❌ [DQ] Data quality checks FAILED")

    for w in report.get("warnings", []):
        print(f"   ⚠️  {w}")

    # Key numbers
    checks = report.get("checks", {})
    stores = sorted({k.split(".")[0] for k in checks})
    for s in stores:
        n  = checks.get(f"{s}.n_customers", "?")
        tr = checks.get(f"{s}.treatment_rate", "?")
        days = checks.get(f"{s}.date_range_days", "?")
        tr_str = f"{tr:.1%}" if isinstance(tr, float) else str(tr)
        print(f"   [{s}] n={n:,}  treatment={tr_str}  days={days}" if isinstance(n, int) else
              f"   [{s}] n={n}  treatment={tr_str}  days={days}")

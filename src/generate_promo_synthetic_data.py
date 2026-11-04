"""Generate synthetic POS + promo history data for training and testing Promo Advisor.

Produces:
  data/promo_synthetic/pos_transactions.csv    — POS transaction records
  data/promo_synthetic/promo_history.jsonl     — simulated past promo runs
  data/promo_synthetic/ground_truth.csv        — per-customer known response labels

Design goals:
  - Different IBM segments have known, distinct response rates per promo type
  - Multiple months of data with 3–4 promo types per month
  - Cannibalization cases: Champions have high baseline (buy regardless)
  - Realistic noise: not all high-uplift customers convert

Segment × Promo response rates (ground truth):
  Segment            | discount | points_vip | retention | notification
  -------------------|----------|------------|-----------|-------------
  champions          |   +1%    |   +6%      |   +2%     |   +3%
  loyal              |   +4%    |   +8%      |   +3%     |   +4%
  potential_loyalist |   +9%    |   +5%      |   +4%     |   +2%
  at_risk            |  +11%    |   +3%      |  +12%     |   +1%
  hibernating        |  +14%    |   +2%      |  +16%     |   +1%
  promising          |   +8%    |   +4%      |   +3%     |   +2%
  new_customer       |   +6%    |   +3%      |   +1%     |   +5%
  cant_lose          |   +3%    |   +7%      |  +10%     |   +3%
  about_to_sleep     |   +7%    |   +3%      |  +9%      |   +2%
  lost_cause         |   +1%    |   +1%      |   +2%     |   +0%

Usage:
  python src/generate_promo_synthetic_data.py
  python src/generate_promo_synthetic_data.py --n-customers 500 --n-months 6 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Ground-truth response lift per segment × promo_type ──────────────────────
# lift = additional conversion probability when receiving this promo type
SEGMENT_PROMO_LIFT: dict[str, dict[str, float]] = {
    "champions":          {"discount": 0.01, "points_vip": 0.06, "retention": 0.02, "notification": 0.03},
    "loyal":              {"discount": 0.04, "points_vip": 0.08, "retention": 0.03, "notification": 0.04},
    "potential_loyalist": {"discount": 0.09, "points_vip": 0.05, "retention": 0.04, "notification": 0.02},
    "at_risk":            {"discount": 0.11, "points_vip": 0.03, "retention": 0.12, "notification": 0.01},
    "hibernating":        {"discount": 0.14, "points_vip": 0.02, "retention": 0.16, "notification": 0.01},
    "promising":          {"discount": 0.08, "points_vip": 0.04, "retention": 0.03, "notification": 0.02},
    "new_customer":       {"discount": 0.06, "points_vip": 0.03, "retention": 0.01, "notification": 0.05},
    "cant_lose":          {"discount": 0.03, "points_vip": 0.07, "retention": 0.10, "notification": 0.03},
    "about_to_sleep":     {"discount": 0.07, "points_vip": 0.03, "retention": 0.09, "notification": 0.02},
    "lost_cause":         {"discount": 0.01, "points_vip": 0.01, "retention": 0.02, "notification": 0.00},
}

# Baseline conversion probability per segment (without any promo)
SEGMENT_BASELINE: dict[str, float] = {
    "champions":          0.72,
    "loyal":              0.60,
    "potential_loyalist": 0.35,
    "at_risk":            0.22,
    "hibernating":        0.08,
    "promising":          0.30,
    "new_customer":       0.25,
    "cant_lose":          0.65,
    "about_to_sleep":     0.28,
    "lost_cause":         0.05,
}

# Average spend per segment (THB)
SEGMENT_AVG_SPEND: dict[str, tuple[float, float]] = {
    "champions":          (350, 100),
    "loyal":              (250,  80),
    "potential_loyalist": (150,  50),
    "at_risk":            (120,  40),
    "hibernating":        ( 90,  30),
    "promising":          (130,  45),
    "new_customer":       (100,  35),
    "cant_lose":          (400, 120),
    "about_to_sleep":     (110,  40),
    "lost_cause":         ( 70,  25),
}

# Promo types available each month (cycle through)
MONTHLY_PROMO_CYCLES: list[list[dict]] = [
    [{"name": "Bundle ลด 15%",          "type": "discount",     "rate": 0.15},
     {"name": "แต้มพิเศษ 2×",           "type": "points_vip",   "rate": 0.05}],
    [{"name": "Welcome Back Offer",      "type": "retention",    "rate": 0.12},
     {"name": "แจ้งเตือนฟรี",           "type": "notification", "rate": 0.00}],
    [{"name": "B1G1 เครื่องดื่ม",       "type": "discount",     "rate": 0.50},
     {"name": "VIP Early Access",        "type": "points_vip",   "rate": 0.05},
     {"name": "Churn Rescue 20%",        "type": "retention",    "rate": 0.20}],
    [{"name": "Weekend Special 10%",     "type": "discount",     "rate": 0.10},
     {"name": "แต้มสะสม VIP",           "type": "points_vip",   "rate": 0.03}],
]

PRODUCTS = [
    ("ครัวซองต์เนยสด",         "ครัวซองต์และเบเกอรี่",  65),
    ("ทาร์ตไข่มัทฉะ",          "ทาร์ตไข่",              49),
    ("โกโก้เข้มข้น",           "มัทฉะและเครื่องดื่ม",   75),
    ("บราวนี่กล่อง 4 ชิ้น",    "บราวนี่",               319),
    ("มัทฉะลาเต้",             "มัทฉะและเครื่องดื่ม",   89),
    ("เค้กชีสสตรอว์เบอร์รี่",  "เค้กและของหวาน",        145),
    ("ครัวซองต์อัลมอนด์",      "ครัวซองต์และเบเกอรี่",  79),
    ("คุกกี้ครีมชีส 6 ชิ้น",   "คุกกี้",                149),
]


def _make_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    segments = list(SEGMENT_BASELINE.keys())
    weights  = [0.08, 0.10, 0.12, 0.10, 0.08, 0.12, 0.12, 0.08, 0.10, 0.10]
    segs = rng.choice(segments, size=n, p=weights)
    avg_spends = np.array([
        max(10.0, rng.normal(SEGMENT_AVG_SPEND[s][0], SEGMENT_AVG_SPEND[s][1]))
        for s in segs
    ]).round(2)
    return pd.DataFrame({
        "customer_id":           [f"cust-{i:04d}" for i in range(n)],
        "segment":               segs,
        "true_baseline_conv":    [SEGMENT_BASELINE[s] for s in segs],
        "avg_spent_ground_truth": avg_spends,
    })


def _simulate_transactions(
    customers: pd.DataFrame,
    start_date: datetime,
    n_months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    end_date = start_date + timedelta(days=30 * n_months)
    bill_id  = 0

    for _, cust in customers.iterrows():
        seg       = cust["segment"]
        avg_spend = float(cust["avg_spent_ground_truth"])
        # Visits per month depends on segment
        visits_per_month = {
            "champions": 8, "loyal": 5, "potential_loyalist": 3,
            "at_risk": 2, "hibernating": 1, "promising": 3,
            "new_customer": 2, "cant_lose": 7, "about_to_sleep": 2,
            "lost_cause": 1,
        }.get(seg, 2)
        total_visits = max(1, int(rng.poisson(visits_per_month * n_months)))

        for _ in range(total_visits):
            ts = start_date + timedelta(seconds=int(rng.uniform(0, (end_date - start_date).total_seconds())))
            bill_id += 1
            bill     = f"bill-{bill_id:06d}"
            n_items  = rng.integers(1, 4)
            for _ in range(n_items):
                prod_idx = rng.integers(0, len(PRODUCTS))
                pname, cat, price = PRODUCTS[prod_idx]
                qty   = rng.integers(1, 3)
                rows.append({
                    "customer_id":   cust["customer_id"],
                    "timestamp":     ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "transaction_id": bill,
                    "invoice_id":    bill,
                    "product_name":  pname,
                    "category":      cat,
                    "quantity":      int(qty),
                    "unit_price":    price,
                    "total_price":   price * qty,
                    "is_treated":    0,
                    "post_promo_visit": 0,
                    "promo_applied": 0,
                    "promo_type":    "none",
                })

    return pd.DataFrame(rows)


def _simulate_promo_responses(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    n_months: int,
    start_date: datetime,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """Assign promos to customers in each month and simulate responses."""
    history_entries = []
    transactions = transactions.copy()

    for month_idx in range(n_months):
        cycle   = MONTHLY_PROMO_CYCLES[month_idx % len(MONTHLY_PROMO_CYCLES)]
        m_start = start_date + timedelta(days=30 * month_idx)
        m_end   = m_start   + timedelta(days=30)

        # Randomly assign ~40% of customers to treatment per promo
        for promo in cycle:
            treated_cids = set(
                customers["customer_id"].sample(frac=0.40, random_state=int(rng.integers(0, 9999))).tolist()
            )

            promo_response_counts = {"sent": 0, "converted": 0}

            for cid in treated_cids:
                seg_row = customers[customers["customer_id"] == cid]
                if seg_row.empty:
                    continue
                seg         = str(seg_row.iloc[0]["segment"])
                baseline    = float(seg_row.iloc[0]["true_baseline_conv"])
                lift        = SEGMENT_PROMO_LIFT.get(seg, {}).get(promo["type"], 0.0)
                conv_prob   = min(1.0, baseline + lift + rng.normal(0, 0.02))
                did_convert = rng.random() < conv_prob

                # Mark transactions in this month as treated + response
                month_mask = (
                    (transactions["customer_id"] == cid) &
                    (pd.to_datetime(transactions["timestamp"]).dt.tz_localize(None) >= m_start) &
                    (pd.to_datetime(transactions["timestamp"]).dt.tz_localize(None) <  m_end)
                )
                transactions.loc[month_mask, "is_treated"]      = 1
                transactions.loc[month_mask, "post_promo_visit"] = int(did_convert)
                transactions.loc[month_mask, "promo_applied"]    = 1
                transactions.loc[month_mask, "promo_type"]       = promo["type"]

                promo_response_counts["sent"]      += 1
                promo_response_counts["converted"] += int(did_convert)

            # Log this promo run as a history entry
            conv_rate = (promo_response_counts["converted"] / max(1, promo_response_counts["sent"]))
            history_entries.append({
                "logged_at":  (m_start + timedelta(days=1)).isoformat(),
                "promo_spec": {
                    "name":          promo["name"],
                    "discount_rate": promo["rate"],
                    "promo_type":    promo["type"],
                    "scope":         "all_items",
                    "category":      None,
                    "items":         [],
                    "target":        "all_customers",
                    "segments":      [],
                    "start_date":    m_start.strftime("%Y-%m-%d"),
                    "end_date":      (m_start + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "force":         False,
                    "tie_threshold": 0.10,
                },
                "summary": {
                    "promo_name":          promo["name"],
                    "promo_discount_rate": promo["rate"],
                    "total_customers":     len(customers),
                    "scenario_C": {
                        "send_count": promo_response_counts["sent"],
                        "profit":     round(promo_response_counts["converted"] *
                                           100.0 * (1 - promo["rate"]) * 0.5, 2),
                    },
                },
            })

    return transactions, history_entries


def _make_ground_truth(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Per-customer summary with true labels for validation."""
    cust_agg = (
        transactions.groupby("customer_id")
        .agg(
            total_txns=("transaction_id", "nunique"),
            total_spend=("total_price", "sum"),
            avg_spend=("total_price", "mean"),
            times_treated=("is_treated", "max"),
            times_converted=("post_promo_visit", "max"),
            promo_types_received=("promo_type", lambda x: list(x[x != "none"].unique())),
        )
        .reset_index()
    )
    return customers.merge(cust_agg, on="customer_id", how="left")


def generate(
    n_customers: int = 300,
    n_months: int = 4,
    seed: int = 42,
    output_dir: str = "data/promo_synthetic",
) -> dict[str, str]:
    """Generate all synthetic datasets. Returns dict of output file paths."""
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    start_date = datetime(2026, 1, 1)

    print(f"\n{'='*56}")
    print(f"  Promo Synthetic Data Generator")
    print(f"{'='*56}")
    print(f"  Customers : {n_customers:,}")
    print(f"  Months    : {n_months}")
    print(f"  Seed      : {seed}")
    print(f"  Output    : {output_dir}/\n")

    customers    = _make_customers(n_customers, rng)
    transactions = _simulate_transactions(customers, start_date, n_months, rng)
    transactions, history_entries = _simulate_promo_responses(
        transactions, customers, n_months, start_date, rng
    )
    ground_truth = _make_ground_truth(customers, transactions)

    # ── Save files ────────────────────────────────────────────────────────────
    pos_path  = os.path.join(output_dir, "pos_transactions.csv")
    gt_path   = os.path.join(output_dir, "ground_truth.csv")
    hist_path = os.path.join(output_dir, "promo_history.jsonl")

    transactions.to_csv(pos_path, index=False)
    ground_truth.to_csv(gt_path,  index=False)
    with open(hist_path, "w", encoding="utf-8") as f:
        for entry in history_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  POS transactions : {len(transactions):,} rows  → {pos_path}")
    print(f"  Unique customers : {transactions['customer_id'].nunique():,}")
    print(f"  Promo history    : {len(history_entries)} entries  → {hist_path}")
    print(f"  Ground truth     : {len(ground_truth):,} rows  → {gt_path}")
    print(f"\n  Segment breakdown:")
    for seg, cnt in customers["segment"].value_counts().items():
        bl  = SEGMENT_BASELINE[seg]
        print(f"    {seg:<22} {cnt:>4,} customers  baseline={bl:.0%}")

    print(f"\n  Promo types in data:")
    for p_type in ["discount", "points_vip", "retention", "notification"]:
        best_seg = max(SEGMENT_PROMO_LIFT, key=lambda s: SEGMENT_PROMO_LIFT[s][p_type])
        best_lift = SEGMENT_PROMO_LIFT[best_seg][p_type]
        print(f"    {p_type:<14} best segment: {best_seg} (lift={best_lift:+.0%})")

    return {"pos": pos_path, "ground_truth": gt_path, "history": hist_path}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-customers", type=int, default=300)
    p.add_argument("--n-months",    type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output-dir",  default="data/promo_synthetic")
    args = p.parse_args(argv)

    generate(
        n_customers=args.n_customers,
        n_months=args.n_months,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Generate synthetic scenario CSVs for testing pipeline coverage.
ครอบคลุม 5 scenarios ที่ขาดใน transactions.csv:

  1. reactivation   — ลูกค้าหายไป 6-18 เดือน + former champions ที่กำลัง slip
  2. new_customers  — new cohort (first purchase 1-60 days ago)
  3. low_season     — Jan-Jun window, post-peak slump
  4. b2b_mix        — wholesale buyers (basket ใหญ่ผิดปกติ) ปน retail
  5. ab_experiment  — 300 คนที่มี random treatment label (discount / none)
"""
import numpy as np
import pandas as pd
import argparse
import os

PRODUCTS = {
    "CERAMIC MUG": 3.95,
    "GIFT BAG LARGE": 2.10,
    "PICTURE FRAME": 5.75,
    "SCENTED CANDLE": 4.50,
    "NOTEBOOK": 3.25,
    "WALL CLOCK": 14.95,
    "STORAGE BOX": 8.75,
    "CUSHION COVER": 6.50,
    "TABLE RUNNER": 9.95,
    "GLASS VASE": 11.25,
}

# Realistic quantity distribution matching transactions.csv (UCI Online Retail)
# Weighted to reflect typical B2C+B2B mix: small retail + bulk orders
_QTY_VALUES  = [1,  2,  3,  4,  6,  8,  10, 12, 16, 20, 24, 36, 48, 96]
_QTY_WEIGHTS = [27, 15, 7,  7,  8,  2,  4,  11, 1,  1,  4,  1,  1,  1]


def _sample_qty(rng, n=1):
    probs = np.array(_QTY_WEIGHTS, dtype=float)
    probs /= probs.sum()
    idx = rng.choice(len(_QTY_VALUES), size=n, p=probs)
    return [_QTY_VALUES[i] for i in idx]


def _rows(rng, cid, timestamps, spend_scale=1.0, label=None):
    """Generate line-item rows for one customer over given timestamps."""
    rows = []
    products = list(PRODUCTS.keys())
    prices = list(PRODUCTS.values())
    for ts in timestamps:
        n_items = int(rng.integers(1, 5))
        qtys = _sample_qty(rng, n_items)
        for i in range(n_items):
            idx = int(rng.integers(0, len(products)))
            qty = int(qtys[i])
            price = round(prices[idx] * qty * spend_scale, 2)
            row = dict(
                transaction_id=f"SYN{abs(hash((cid, str(ts), i))) % 1_000_000:07d}",
                customer_id=cid,
                timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                product_name=products[idx],
                quantity=qty,
                total_price=price,
            )
            if label is not None:
                row.update(is_treated=label["is_treated"],
                           post_promo_visit=label["post_promo_visit"],
                           promo_type=label["promo_type"])
            rows.append(row)
    return rows


def _timestamps(rng, base_date, n_visits, jitter_days=180):
    """Return n_visits sorted Timestamps near base_date ± jitter."""
    offsets = rng.integers(-jitter_days, 0, size=n_visits)
    return sorted([base_date + pd.Timedelta(days=int(d),
                                            hours=int(rng.integers(8, 21)),
                                            minutes=int(rng.integers(0, 60)))
                   for d in offsets])


# ── Scenario 1: Reactivation ──────────────────────────────────────────────────
def scenario_reactivation(seed=1, n_dormant=250, n_slipping=80, n_anchor=20):
    """
    - dormant     : last purchase 180-540 days ago, 1-4 visits total
    - slipping    : ex-champion, last purchase 90-180 days ago, frequency dropping
    - anchor      : small group of recent actives to calibrate stock (spread over 6 months)
    Reference date = 2024-06-01 (post-peak slump)
    anchor customers' visits are spread over 180 days to avoid recent stock-depletion spike.
    """
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2024-06-01")
    rows = []

    for i in range(n_dormant):
        cid = f"REAC_DORM_{i:04d}"
        last_offset = -int(rng.integers(180, 540))
        last_ts = ref + pd.Timedelta(days=last_offset)
        n_visits = int(rng.integers(1, 4))
        tss = _timestamps(rng, last_ts, n_visits, jitter_days=min(120, abs(last_offset)-1))
        rows += _rows(rng, cid, tss)

    for i in range(n_slipping):
        cid = f"REAC_SLIP_{i:04d}"
        last_offset = -int(rng.integers(90, 180))
        last_ts = ref + pd.Timedelta(days=last_offset)
        n_visits = int(rng.integers(4, 10))
        tss = _timestamps(rng, last_ts, n_visits, jitter_days=300)
        spend = float(rng.uniform(1.5, 3.0))
        rows += _rows(rng, cid, tss, spend_scale=spend)

    # Anchor customers: few visits spread over 180 days (not concentrated in last 14 days)
    for i in range(n_anchor):
        cid = f"REAC_ACT_{i:04d}"
        n_visits = int(rng.integers(3, 7))
        tss = _timestamps(rng, ref, n_visits, jitter_days=180)
        rows += _rows(rng, cid, tss)

    return pd.DataFrame(rows)


# ── Scenario 2: New customers ─────────────────────────────────────────────────
def scenario_new_customers(seed=2, n_new=200, n_oneshot=80, n_champion=50):
    """
    - new         : first purchase 1-60 days ago, 1-3 visits
    - oneshot     : bought once, never returned (high churn risk)
    - champion    : established Champions as anchor
    Reference date = 2024-11-01
    """
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2024-11-01")
    rows = []

    for i in range(n_new):
        cid = f"NEW_CUST_{i:04d}"
        first_offset = -int(rng.integers(1, 60))
        first_ts = ref + pd.Timedelta(days=first_offset)
        n_visits = int(rng.integers(1, 4))
        tss = _timestamps(rng, first_ts, n_visits, jitter_days=abs(first_offset))
        rows += _rows(rng, cid, tss)

    for i in range(n_oneshot):
        cid = f"NEW_ONE_{i:04d}"
        first_offset = -int(rng.integers(30, 90))
        first_ts = ref + pd.Timedelta(days=first_offset)
        rows += _rows(rng, cid, [first_ts])

    for i in range(n_champion):
        cid = f"NEW_CHAMP_{i:04d}"
        # Use 4-10 visits (not 15-30) to avoid a bimodal gap between new (1-3) and
        # champion (15-30) customers, which causes artificially high frequency PSI.
        n_visits = int(rng.integers(4, 11))
        tss = _timestamps(rng, ref, n_visits, jitter_days=365)
        spend = float(rng.uniform(1.8, 3.5))
        rows += _rows(rng, cid, tss, spend_scale=spend)

    return pd.DataFrame(rows)


# ── Scenario 3: Low season ────────────────────────────────────────────────────
def scenario_low_season(seed=3, n_customers=350):
    """Jan-Jun window only — post-peak slump, 40% lower volume."""
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2024-06-30")
    rows = []
    for i in range(n_customers):
        cid = f"LOW_SEAS_{i:04d}"
        n_visits = int(rng.integers(2, 8))
        tss = _timestamps(rng, ref, n_visits, jitter_days=180)  # Jan-Jun
        rows += _rows(rng, cid, tss, spend_scale=float(rng.uniform(0.6, 1.1)))
    return pd.DataFrame(rows)


# ── Scenario 4: B2B mix ───────────────────────────────────────────────────────
def scenario_b2b_mix(seed=4, n_b2b=60, n_b2c=250):
    """Wholesale (B2B) buyers with basket 5-20x larger than retail."""
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2024-11-01")
    rows = []

    for i in range(n_b2b):
        cid = f"B2B_{i:04d}"
        n_visits = int(rng.integers(3, 10))
        tss = _timestamps(rng, ref, n_visits, jitter_days=300)
        spend = float(rng.uniform(8, 20))  # B2B: bulk orders
        rows += _rows(rng, cid, tss, spend_scale=spend)

    for i in range(n_b2c):
        cid = f"B2C_{i:04d}"
        n_visits = int(rng.integers(3, 12))
        tss = _timestamps(rng, ref, n_visits, jitter_days=300)
        rows += _rows(rng, cid, tss)

    return pd.DataFrame(rows)


# ── Scenario 5: A/B experiment ────────────────────────────────────────────────
def scenario_ab_experiment(seed=5, n_treated=150, n_control=150):
    """
    Randomised treatment assignment:
    - treated: received a 15% discount campaign
    - control: no campaign
    Treated group has higher conversion rate (+18% lift)
    """
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp("2024-11-01")
    rows = []

    for i in range(n_treated):
        cid = f"AB_TREAT_{i:04d}"
        converted = int(rng.random() < 0.55)   # 55% conversion with discount
        n_visits = int(rng.integers(2, 8))
        tss = _timestamps(rng, ref, n_visits, jitter_days=180)
        label = dict(is_treated=1, post_promo_visit=converted, promo_type="discount_15pct")
        rows += _rows(rng, cid, tss, label=label)

    for i in range(n_control):
        cid = f"AB_CTRL_{i:04d}"
        converted = int(rng.random() < 0.37)   # 37% baseline conversion
        n_visits = int(rng.integers(2, 8))
        tss = _timestamps(rng, ref, n_visits, jitter_days=180)
        label = dict(is_treated=0, post_promo_visit=converted, promo_type="none")
        rows += _rows(rng, cid, tss, label=label)

    return pd.DataFrame(rows)


SCENARIOS = {
    "reactivation": scenario_reactivation,
    "new_customers": scenario_new_customers,
    "low_season": scenario_low_season,
    "b2b_mix": scenario_b2b_mix,
    "ab_experiment": scenario_ab_experiment,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                   choices=list(SCENARIOS), metavar="SCENARIO")
    p.add_argument("--out-dir", default="data/scenarios")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.scenarios:
        df = SCENARIOS[name]()
        path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"✅ {name:20s} → {path}  ({len(df):,} rows, {df['customer_id'].nunique()} customers)")


if __name__ == "__main__":
    main()

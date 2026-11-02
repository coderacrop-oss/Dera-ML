"""
Generate synthetic POS data specifically designed to test Layer 3 (Context Detection).

What this dataset creates:
  1. Peak-hour clustering  — 70% of transactions land in 17:00-21:00 (CONFIG peak_hour_start/end)
                             so Layer 3 should label most rows 'Peak', not 'Off-Peak'
  2. Stockout products     — 3 "hot" products receive a 6× demand surge in the final 14 days
                             → remaining_stock falls below threshold → in_stock = False
                             → customers whose top_product is one of these get
                               "Do Nothing (Out of Stock)" from Layer 4
  3. Stable products       — 7 products with uniform demand throughout; should stay in_stock
  4. Treatment labels      — ~40% of customers flagged is_treated=1 so Layer 4 X-learner
                             can split into meaningful treatment/control groups

Output: data/synthetic_layer3_test.csv  (~4,800 rows, 500 customers)

Run with:
    cd /Users/ninakim/Documents/ml-campaign-engine
    python data/generate_layer3_synthetic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_CUSTOMERS     = 500
N_DAYS          = 90          # 90-day study period
ANCHOR          = pd.Timestamp("2025-08-01 00:00:00")  # study start (naive → Bangkok tz added later)
PEAK_HOURS      = list(range(17, 22))                  # 17-21 inclusive (matches CONFIG)
PEAK_WEIGHT     = 0.70                                 # 70% of transactions in peak hours
RESTOCK_DAYS    = 14                                   # must match CONFIG["stock_restock_cycle_days"]
SURGE_FACTOR    = 6.0                                  # hot products: 6× demand in last 14 days
TREATMENT_RATE  = 0.40

# Products: (name, base_unit_price, base_daily_transactions_per_product)
PRODUCTS_NORMAL = [
    ("Classic Home Decor Set",       45.00),
    ("Ceramic Mug Collection",       12.50),
    ("Linen Throw Pillow",           28.00),
    ("Bamboo Storage Basket",        22.00),
    ("Glass Vase Set",               35.00),
    ("Wooden Picture Frame",         18.00),
    ("Cotton Napkin Set",             9.50),
]
# Hot products — will have surge in last 14 days to trigger stockout
PRODUCTS_HOT = [
    ("Special Edition Gift Set",     89.00),
    ("Holiday Hamper Box",          120.00),
    ("Premium Scented Candle Bundle", 65.00),
]
ALL_PRODUCTS = PRODUCTS_NORMAL + PRODUCTS_HOT
HOT_NAMES    = {p[0] for p in PRODUCTS_HOT}

# ── Helpers ──────────────────────────────────────────────────────────────────

def random_hour(n, peak=True):
    """Sample n hours: peak=True → 17-21 (70% prob), else rest of day."""
    off_peak_hours = [h for h in range(24) if h not in PEAK_HOURS]
    if peak:
        return rng.choice(PEAK_HOURS, size=n)
    else:
        return rng.choice(off_peak_hours, size=n)


def make_transactions(customer_ids, treated_set):
    rows = []
    tx_id = 1

    for cid in customer_ids:
        # Each customer buys 5-20 times across the study period
        n_purchases = int(rng.integers(5, 21))
        for _ in range(n_purchases):
            day_offset = int(rng.integers(0, N_DAYS))
            is_last_14 = day_offset >= (N_DAYS - RESTOCK_DAYS)

            # Choose product — hot products are rarely purchased except in the surge window
            product_weights = []
            for pname, _ in ALL_PRODUCTS:
                if pname in HOT_NAMES:
                    w = SURGE_FACTOR if is_last_14 else 0.1   # almost no sales before surge
                else:
                    w = 1.0
                product_weights.append(w)
            product_weights = np.array(product_weights, dtype=float)
            product_weights /= product_weights.sum()

            prod_idx       = int(rng.choice(len(ALL_PRODUCTS), p=product_weights))
            prod_name, base_price = ALL_PRODUCTS[prod_idx]

            # Hour: 70% peak, 30% off-peak
            if rng.random() < PEAK_WEIGHT:
                hour = int(rng.choice(PEAK_HOURS))
            else:
                off_peak_hours = [h for h in range(24) if h not in PEAK_HOURS]
                hour = int(rng.choice(off_peak_hours))

            minute    = int(rng.integers(0, 60))
            ts        = ANCHOR + pd.Timedelta(days=day_offset, hours=hour, minutes=minute)
            quantity  = int(rng.integers(1, 4))
            unit_price = round(float(base_price * rng.uniform(0.95, 1.05)), 2)
            total     = round(unit_price * quantity, 2)

            # Treatment & conversion labels
            is_treated = 1 if cid in treated_set else 0
            # Treated customers who buy in peak hours convert slightly more
            if is_treated and hour in PEAK_HOURS:
                converts = 1 if rng.random() < 0.62 else 0
            elif is_treated:
                converts = 1 if rng.random() < 0.48 else 0
            else:
                converts = 1 if rng.random() < 0.38 else 0

            # Promo type
            if is_treated:
                if rng.random() < 0.30:
                    promo = "coupon"
                elif rng.random() < 0.50:
                    promo = "discount"
                else:
                    promo = "none"
            else:
                promo = "none"

            rows.append({
                "transaction_id":  f"TX{tx_id:06d}",
                "invoice_id":       f"INV{tx_id:06d}",
                "customer_id":      str(cid),
                "product_name":     prod_name,
                "timestamp":        ts,
                "quantity":         quantity,
                "unit_price":       unit_price,
                "total_price":      total,
                "is_treated":       is_treated,
                "post_promo_visit": converts,
                "promo_type":       promo,
            })
            tx_id += 1

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

customer_ids = list(range(1001, 1001 + N_CUSTOMERS))
treated_set  = set(rng.choice(customer_ids,
                               size=int(N_CUSTOMERS * TREATMENT_RATE),
                               replace=False).tolist())

print(f"Generating transactions for {N_CUSTOMERS} customers...")
df = make_transactions(customer_ids, treated_set)
df = df.sort_values("timestamp").reset_index(drop=True)

# ── Diagnostics ───────────────────────────────────────────────────────────────
print(f"\nTotal rows       : {len(df):,}")
print(f"Unique customers : {df['customer_id'].nunique():,}")
print(f"Date range       : {df['timestamp'].min()} → {df['timestamp'].max()}")

hour_dist = df['timestamp'].dt.hour
peak_pct  = hour_dist.isin(PEAK_HOURS).mean() * 100
print(f"\nPeak-hour %      : {peak_pct:.1f}%  (expected ~{PEAK_WEIGHT*100:.0f}%)")

# Layer 3 stockout preview
n_days_total = (df['timestamp'].max() - df['timestamp'].min()).days
cutoff = df['timestamp'].max() - pd.Timedelta(days=RESTOCK_DAYS)
print(f"\n--- Stockout preview (restock_cycle={RESTOCK_DAYS} days, buffer=1.5) ---")
for pname, _ in ALL_PRODUCTS:
    total_qty   = df.loc[df['product_name'] == pname, 'quantity'].sum()
    daily_rate  = total_qty / max(1, n_days_total)
    initial_est = int(daily_rate * RESTOCK_DAYS * 1.5)  # matches CONFIG stock_buffer_multiplier
    recent_qty  = df.loc[(df['product_name'] == pname) &
                         (df['timestamp'] >= cutoff), 'quantity'].sum()
    remaining   = max(0, initial_est - recent_qty)
    low_thresh  = max(int(initial_est * 0.20), 5)  # matches Layer 3 logic
    in_stock    = remaining > low_thresh
    hot_flag    = " ← HOT (should be OUT)" if pname in HOT_NAMES else ""
    print(f"  {'OUT' if not in_stock else ' OK '} | {pname:<40} "
          f"init={initial_est:>5} recent={int(recent_qty):>5} remain={remaining:>5}{hot_flag}")

print(f"\nTreatment rate   : {df['is_treated'].mean():.1%}")
print(f"Conversion (treated) : {df[df['is_treated']==1]['post_promo_visit'].mean():.1%}")
print(f"Conversion (control) : {df[df['is_treated']==0]['post_promo_visit'].mean():.1%}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "synthetic_layer3_test.csv"
df.to_csv(out_path, index=False)
print(f"\n✅ Saved → {out_path}  ({len(df):,} rows)")
print("\nRun the pipeline with:")
print("  cd src && python csv_pipeline_engine.py ../data/synthetic_layer3_test.csv")

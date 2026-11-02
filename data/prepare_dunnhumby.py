"""
Prepare Dunnhumby - The Complete Journey dataset for ml-campaign-engine pipeline.

Joins:
  transaction_data + product → product_name, invoice_id, customer_id, timestamp, quantity, unit_price
  campaign_table + coupon_redempt → is_treated, post_promo_visit, promo_type
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent / "dunnhumby"
OUT  = Path(__file__).parent / "dunnhumby_transactions.csv"

print("Loading files...")
txn     = pd.read_csv(BASE / "transaction_data.csv")
product = pd.read_csv(BASE / "product.csv")
camp_tbl = pd.read_csv(BASE / "campaign_table.csv")
redempt  = pd.read_csv(BASE / "coupon_redempt.csv")

# ── 1. Merge product name ────────────────────────────────────────────────────
product["product_name"] = (
    product["COMMODITY_DESC"].str.title().str.strip() + " - " +
    product["SUB_COMMODITY_DESC"].str.title().str.strip()
)
txn = txn.merge(product[["PRODUCT_ID", "product_name"]], on="PRODUCT_ID", how="left")
txn["product_name"] = txn["product_name"].fillna("General Merchandise")

# ── 2. Build timestamp from DAY + TRANS_TIME ────────────────────────────────
# DAY = days since study start (day 1 = 2019-01-01 arbitrary anchor)
anchor = pd.Timestamp("2024-01-01")
txn["transaction_date"] = anchor + pd.to_timedelta(txn["DAY"] - 1, unit="D")
# TRANS_TIME is HHMM integer e.g. 1631 = 16:31
txn["hour"]   = (txn["TRANS_TIME"] // 100).clip(0, 23)
txn["minute"] = (txn["TRANS_TIME"] %  100).clip(0, 59)
txn["transaction_date"] = txn["transaction_date"] + pd.to_timedelta(txn["hour"], unit="h") + pd.to_timedelta(txn["minute"], unit="m")

# ── 3. Unit price: SALES_VALUE / QUANTITY (avoid div/0) ─────────────────────
txn["unit_price"] = (txn["SALES_VALUE"] / txn["QUANTITY"].replace(0, 1)).round(4)
txn = txn[txn["SALES_VALUE"] > 0]  # drop returns/voids

# ── 4. Rename core fields ────────────────────────────────────────────────────
txn = txn.rename(columns={
    "household_key": "customer_id",
    "BASKET_ID":     "invoice_id",
    "QUANTITY":      "quantity",
    "SALES_VALUE":   "total_price",
})

# ── 5. Build treatment labels ────────────────────────────────────────────────
# camp_tbl: which household received which campaign
# coupon_redempt: which household redeemed a coupon on which day

# Mark treated households (received any campaign)
treated_hh = set(camp_tbl["household_key"].astype(str))

txn["customer_id"] = txn["customer_id"].astype(str)
txn["is_treated"] = txn["customer_id"].isin(treated_hh).astype(int)

# post_promo_visit = purchased in the latter half of the study period
# Proxy: customers who kept purchasing after week 52 (mid-point of 117 weeks)
# This gives a realistic conversion signal across both treated and control
mid_week = txn["WEEK_NO"].median()
txn["post_promo_visit"] = (txn["WEEK_NO"] > mid_week).astype(int)

# promo_type: coupon → coupon, retail discount → discount, else none
def promo_type(row):
    if row["COUPON_DISC"] < 0 or row["COUPON_MATCH_DISC"] < 0:
        return "coupon"
    if row["RETAIL_DISC"] < 0:
        return "discount"
    return "none"

print("Assigning promo_type...")
txn["promo_type"] = txn.apply(promo_type, axis=1)

# ── 6. Select final columns ──────────────────────────────────────────────────
out = txn[[
    "customer_id", "invoice_id", "transaction_date",
    "product_name", "quantity", "unit_price", "total_price",
    "is_treated", "post_promo_visit", "promo_type"
]].copy()

out = out.sort_values("transaction_date").reset_index(drop=True)

print(f"Saving {len(out):,} rows, {out['customer_id'].nunique():,} customers...")
out.to_csv(OUT, index=False)
print(f"✅ Saved to {OUT}")
print(f"\nSample:")
print(out.head(3).to_string())
print(f"\nPromo distribution:\n{out['promo_type'].value_counts()}")
print(f"\nTreatment rate: {out['is_treated'].mean():.1%}")
print(f"Post-promo visit rate (treated): {out[out['is_treated']==1]['post_promo_visit'].mean():.1%}")
print(f"Post-promo visit rate (control): {out[out['is_treated']==0]['post_promo_visit'].mean():.1%}")

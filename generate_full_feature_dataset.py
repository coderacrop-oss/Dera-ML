"""
generate_full_feature_dataset.py
─────────────────────────────────
สร้าง synthetic POS dataset ที่ออกแบบให้ exercise ทุก feature ใน 1 run

Segments (500 customers, 90 days):
  A. VIP Peak        (50)  — ซื้อช่วง peak 18-21h, AOV สูง, rfm_decile ≥ 8
                             → ได้: VIP Reward / VIP Early Access
  B. Off-Peak Mid    (80)  — ซื้อช่วง off-peak 9-12h, AOV กลาง, rfm_decile 4-7
                             → ได้: Off-Peak Traffic Driver / Off-Peak VIP Points
  C. Bundle Buyers   (70)  — ซื้อ product pairs ซ้ำๆ สูง (Apriori rules)
                             → ได้: Push Promotion: Bundle
  D. Festival Spike  (30)  — spike transactions ช่วง campaign week, rfm ≥ 7
                             → ได้: Standard Retention (Festival Guard)
  E. Low Basket      (50)  — AOV < 100 → Do Nothing (Low Basket Size)
  F. Out of Stock    (60)  — ซื้อ product "HotItem-OOS" เยอะมาก → L3 flags out of stock
                             → Do Nothing (Out of Stock)
  G. CFO Block       (80)  — uplift ต่ำ / margin ไม่ผ่าน → Do Nothing (CFO Guardrail)
  H. Control Flow    (80)  — ซื้อ random ปกติ — expected: Standard Retention หรือ Guardrail

Peak hours engineered: 18, 19, 20, 21 (volume >> mean + 0.5*std)
Out-of-stock trigger: product "HotItem-OOS" — demand > initial_stock_est in last 14 days
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

OUT_PATH = "data/synthetic_full_features.csv"
LABELS_PATH = "data/synthetic_full_features_labels.csv"

# ── Config mirrors business_config.py thresholds ────────────────────────────
PEAK_HOURS    = [18, 19, 20, 21]      # will be >> mean+0.5*std
OFFPEAK_HOURS = [9, 10, 11, 12]
MID_HOURS     = [13, 14, 15, 16, 17]  # medium volume

LOW_BASKET_THRESHOLD = 100   # AOV < 100 → Do Nothing (Low Basket)
COGS_RATE     = 0.50
MIN_MARGIN    = 0.15
DISCOUNT_RATE = 0.15

# Products
PEAK_PRODUCTS    = ["GiftSet-Premium", "HomeDecor-Luxury", "Candle-Collection"]
OFFPEAK_PRODUCTS = ["Wall-Art-Print", "Ceramic-Mug", "Photo-Frame"]
BUNDLE_A         = ["Cushion-Cover", "Throw-Blanket"]   # ← Apriori pair
BUNDLE_B         = ["Table-Lamp", "Lampshade"]          # ← Apriori pair
FESTIVAL_PRODS   = ["Christmas-Ornament", "Gift-Wrap-Set"]
LOW_PRODUCTS     = ["Sticker-Pack", "Bookmark"]         # cheap → low AOV
OOS_PRODUCT      = "HotItem-OOS"                        # engineered stockout
GENERIC_PRODS    = ["Storage-Box", "Plant-Pot", "Door-Mat", "Coaster-Set", "Mirror-Frame"]

START_DATE = datetime(2026, 8, 1)
END_DATE   = datetime(2026, 10, 29)  # 90 days


def rand_dt(rng, start, end, hours):
    """Random datetime within window, constrained to given hours."""
    delta = (end - start).days
    d = start + timedelta(days=int(rng.integers(0, delta)))
    h = int(rng.choice(hours))
    m = int(rng.integers(0, 60))
    return d.replace(hour=h, minute=m, second=0)


def make_transaction(tx_id, customer_id, dt, product, qty, unit_price,
                     is_treated=0, post_promo_visit=0, promo_type="none"):
    return {
        "transaction_id":  tx_id,
        "customer_id":     customer_id,
        "timestamp":       dt.strftime("%Y-%m-%d %H:%M:%S"),
        "product_name":    product,
        "quantity":        qty,
        "unit_price":      unit_price,
        "is_treated":      is_treated,
        "post_promo_visit": post_promo_visit,
        "promo_type":      promo_type,
    }


def generate(seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = []
    tx_counter = [0]

    def next_tx():
        tx_counter[0] += 1
        return f"T{tx_counter[0]:06d}"

    # ── ฟังก์ชันช่วย: เพิ่มธุรกรรมให้ customer ────────────────────────────
    def add_txs(cid, n_visits, hours, products, price_range, is_treated, ppv, promo,
                multi_item=False, force_end=False):
        """Add n_visits transactions for a customer."""
        for v in range(n_visits):
            if force_end:
                # ซื้อใน 14 วันสุดท้าย (สำหรับ OOS trigger)
                dt = rand_dt(rng, END_DATE - timedelta(days=14), END_DATE, hours)
            else:
                dt = rand_dt(rng, START_DATE, END_DATE, hours)
            product = products[int(rng.integers(0, len(products)))]
            qty = int(rng.integers(1, 4))
            price = round(float(rng.uniform(*price_range)), 2)
            rows.append(make_transaction(next_tx(), cid, dt, product, qty, price,
                                         is_treated, ppv, promo))
            if multi_item:
                # เพิ่ม item ที่สองใน same basket (same timestamp) เพื่อ Apriori
                product2 = products[(products.index(product) + 1) % len(products)]
                rows.append(make_transaction(next_tx(), cid, dt, product2, 1, price * 0.8,
                                             is_treated, ppv, promo))

    # ─────────────────────────────────────────────────────────────────────────
    # Segment A: VIP Peak (50 customers)
    # Goal: rfm_decile ≥ 8 (high frequency + recency + monetary)
    #       peak hours 18-21, AOV 300-600 → VIP Reward / VIP Early Access
    # Treatment: 60% treated (high response in peak)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(50):
        cid = f"VIP-{i:03d}"
        treated = 1 if rng.random() < 0.6 else 0
        ppv = 1 if treated and rng.random() < 0.75 else (1 if not treated and rng.random() < 0.40 else 0)
        promo = "discount" if treated else "none"
        add_txs(cid, n_visits=int(rng.integers(12, 20)), hours=PEAK_HOURS,
                products=PEAK_PRODUCTS, price_range=(300, 600),
                is_treated=treated, ppv=ppv, promo=promo)

    # ─────────────────────────────────────────────────────────────────────────
    # Segment B: Off-Peak Mid-tier (80 customers)
    # Goal: majority off-peak context, rfm_decile 4-7, AOV 150-400
    #       → Off-Peak Traffic Driver / Off-Peak VIP Points
    # Treatment: 40% treated
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(80):
        cid = f"OFP-{i:03d}"
        treated = 1 if rng.random() < 0.4 else 0
        ppv = 1 if treated and rng.random() < 0.58 else (1 if not treated and rng.random() < 0.30 else 0)
        promo = "coupon" if treated else "none"
        # 80% ของ visits อยู่ช่วง off-peak, 20% mid-hours
        n_off = int(rng.integers(6, 12))
        n_mid = int(rng.integers(1, 3))
        add_txs(cid, n_visits=n_off, hours=OFFPEAK_HOURS,
                products=OFFPEAK_PRODUCTS, price_range=(150, 400),
                is_treated=treated, ppv=ppv, promo=promo)
        add_txs(cid, n_visits=n_mid, hours=MID_HOURS,
                products=OFFPEAK_PRODUCTS, price_range=(150, 400),
                is_treated=treated, ppv=ppv, promo=promo)

    # ─────────────────────────────────────────────────────────────────────────
    # Segment C: Bundle Buyers (70 customers)
    # Goal: ซื้อ BUNDLE_A pair และ BUNDLE_B pair บ่อยในรอบเดียว
    #       Apriori จะเจอ rules: Cushion-Cover → Throw-Blanket, Table-Lamp → Lampshade
    # Treatment: 40% treated
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(70):
        cid = f"BND-{i:03d}"
        treated = 1 if rng.random() < 0.4 else 0
        ppv = 1 if treated and rng.random() < 0.62 else (1 if not treated and rng.random() < 0.35 else 0)
        promo = "discount" if treated else "none"
        # ซื้อ bundle pair ทั้งคู่ในรอบเดียว (multi_item=True)
        add_txs(cid, n_visits=int(rng.integers(5, 10)),
                hours=PEAK_HOURS + OFFPEAK_HOURS,
                products=BUNDLE_A, price_range=(200, 450),
                is_treated=treated, ppv=ppv, promo=promo, multi_item=True)
        add_txs(cid, n_visits=int(rng.integers(3, 7)),
                hours=PEAK_HOURS + OFFPEAK_HOURS,
                products=BUNDLE_B, price_range=(180, 380),
                is_treated=treated, ppv=ppv, promo=promo, multi_item=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Segment D: Festival Spike (30 customers)
    # Goal: transaction ราคาสูงมาก (spike) ช่วง campaign week + rfm_seg ≥ 7
    #       is_campaign_spike = 1 เมื่อ total_price > mean + 2.5*std
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(30):
        cid = f"FST-{i:03d}"
        treated = 1 if rng.random() < 0.5 else 0
        ppv = 1 if treated and rng.random() < 0.65 else (1 if not treated and rng.random() < 0.40 else 0)
        promo = "coupon" if treated else "none"
        # Normal transactions
        add_txs(cid, n_visits=int(rng.integers(6, 10)),
                hours=PEAK_HOURS, products=FESTIVAL_PRODS, price_range=(200, 400),
                is_treated=treated, ppv=ppv, promo=promo)
        # Spike transactions (very high price → triggers is_campaign_spike)
        for _ in range(int(rng.integers(2, 5))):
            dt = rand_dt(rng, START_DATE, END_DATE, PEAK_HOURS)
            rows.append(make_transaction(
                next_tx(), cid, dt,
                product=FESTIVAL_PRODS[0],
                qty=int(rng.integers(3, 8)),
                unit_price=round(float(rng.uniform(1500, 3000)), 2),  # ← spike price
                is_treated=treated, post_promo_visit=ppv, promo_type=promo
            ))

    # ─────────────────────────────────────────────────────────────────────────
    # Segment E: Low Basket (50 customers)
    # Goal: AOV < 100 → Do Nothing (Low Basket Size)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(50):
        cid = f"LOW-{i:03d}"
        treated = 1 if rng.random() < 0.4 else 0
        ppv = 1 if treated and rng.random() < 0.30 else 0
        add_txs(cid, n_visits=int(rng.integers(3, 7)),
                hours=PEAK_HOURS + OFFPEAK_HOURS,
                products=LOW_PRODUCTS, price_range=(10, 60),   # ← AOV < 100
                is_treated=treated, ppv=ppv, promo="none")

    # ─────────────────────────────────────────────────────────────────────────
    # Segment F: Out-of-Stock (60 customers)
    # Goal: ซื้อ OOS_PRODUCT เยอะใน 14 วันสุดท้ายมากๆ
    #       → L3 คำนวณ remaining_stock ≤ initial_stock_est * 0.20 → in_stock = False
    #       → Do Nothing (Out of Stock)
    #
    # Logic: initial_stock_est = daily_rate * 14 * 1.5
    #        recent_demand = qty ใน 14 วันสุดท้าย
    #        remaining = initial - recent
    #        in_stock = remaining > max(5, initial * 0.20)
    #        → ต้องการ recent_demand > initial * 0.80
    #        → recent_demand > (daily_rate * 14 * 1.5) * 0.80
    #        → เพิ่ม demand ใน 14 วันสุดท้ายให้ >> daily_rate * 14
    # ─────────────────────────────────────────────────────────────────────────
    # สร้าง background transactions ปกติ (เพื่อ establish daily_rate)
    for i in range(60):
        cid = f"OOS-{i:03d}"
        treated = 1 if rng.random() < 0.35 else 0
        ppv = 1 if treated and rng.random() < 0.35 else 0
        # Background: small qty ตลอด 76 วันแรก
        add_txs(cid, n_visits=int(rng.integers(3, 6)),
                hours=PEAK_HOURS + OFFPEAK_HOURS,
                products=[OOS_PRODUCT], price_range=(200, 350),
                is_treated=treated, ppv=ppv, promo="none")

    # เพิ่ม demand surge ใน 14 วันสุดท้ายสำหรับ OOS product (ซื้อหลายรายการต่อครั้ง)
    # ต้องการ recent_demand >> daily_rate * 14
    # สร้าง spike demand จาก customers เพิ่มเติม (ไม่ใช่ customer เดิม) เพื่อให้ total qty สูง
    n_oos_customers = 60
    for j in range(200):  # 200 additional surge transactions from OOS customers
        cid = f"OOS-{int(rng.integers(0, n_oos_customers)):03d}"
        dt = rand_dt(rng, END_DATE - timedelta(days=14), END_DATE,
                     PEAK_HOURS + OFFPEAK_HOURS)
        qty = int(rng.integers(8, 20))   # ← qty สูงมาก ทำให้ recent_demand >> initial
        rows.append(make_transaction(
            next_tx(), cid, dt, OOS_PRODUCT, qty,
            round(float(rng.uniform(200, 350)), 2),
            0, 0, "none"
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Segment G: CFO Block (80 customers)
    # Goal: uplift ต่ำ / เพื่อให้ guardrail block
    #       - high prob_control (ซื้อบ่อยอยู่แล้ว) → incremental ≈ 0 → guardrail block
    #       - rfm ปานกลาง, treated/not-treated มั่ว → model ไม่ถนัดแยก
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(80):
        cid = f"CFO-{i:03d}"
        # is_treated random ไม่มี signal → uplift ≈ 0
        treated = int(rng.random() < 0.5)
        # response rate เท่ากัน treated/control → ไม่มี treatment effect
        ppv = 1 if rng.random() < 0.45 else 0   # same rate treated+control
        add_txs(cid, n_visits=int(rng.integers(4, 8)),
                hours=PEAK_HOURS + OFFPEAK_HOURS,
                products=GENERIC_PRODS, price_range=(120, 280),
                is_treated=treated, ppv=ppv, promo="none")

    # ─────────────────────────────────────────────────────────────────────────
    # Segment H: General (80 customers)
    # Mix ปกติ — expected: Standard Retention หรือ Guardrail
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(80):
        cid = f"GEN-{i:03d}"
        treated = 1 if rng.random() < 0.4 else 0
        ppv = 1 if treated and rng.random() < 0.50 else (1 if not treated and rng.random() < 0.30 else 0)
        promo = random.choice(["coupon", "discount"]) if treated else "none"
        add_txs(cid, n_visits=int(rng.integers(3, 9)),
                hours=PEAK_HOURS + MID_HOURS + OFFPEAK_HOURS,
                products=GENERIC_PRODS, price_range=(100, 350),
                is_treated=treated, ppv=ppv, promo=promo)

    # ─────────────────────────────────────────────────────────────────────────
    # เพิ่ม background volume ให้ off-peak hours มี volume ปานกลาง
    # และ peak hours 18-21 มี volume สูงชัดเจน (เพื่อ L3 detection)
    # ─────────────────────────────────────────────────────────────────────────
    # เพิ่ม peak hour transactions เพิ่มเติมให้ชัด
    for _ in range(400):
        cid = f"VIP-{int(rng.integers(0, 50)):03d}"
        dt = rand_dt(rng, START_DATE, END_DATE, PEAK_HOURS)
        rows.append(make_transaction(
            next_tx(), cid, dt,
            PEAK_PRODUCTS[int(rng.integers(0, len(PEAK_PRODUCTS)))],
            int(rng.integers(1, 3)),
            round(float(rng.uniform(300, 600)), 2)
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Build DataFrame + total_price
    # ─────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["total_price"] = df["quantity"] * df["unit_price"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── สรุป stats ──────────────────────────────────────────────────────────
    n_cust = df["customer_id"].nunique()
    n_tx   = len(df)
    date_range = f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}"
    treated_pct = df.drop_duplicates("customer_id")["is_treated"].mean() * 100

    # Hourly volume (preview)
    df_ts = pd.to_datetime(df["timestamp"])
    hourly = df_ts.dt.hour.value_counts().sort_index()
    mean_v = hourly.mean()
    std_v  = hourly.std()
    thresh = mean_v + 0.5 * std_v
    peak_detected = sorted(hourly[hourly > thresh].index.tolist())

    # Product volume (สำหรับ OOS check preview)
    oos_total = df[df["product_name"] == OOS_PRODUCT]["quantity"].sum()
    oos_last14 = df[
        (df["product_name"] == OOS_PRODUCT) &
        (pd.to_datetime(df["timestamp"]) >= pd.to_datetime(END_DATE) - timedelta(days=14))
    ]["quantity"].sum()

    print(f"\n{'='*60}")
    print(f"  Synthetic Full-Feature Dataset")
    print(f"{'='*60}")
    print(f"  Customers  : {n_cust:,}")
    print(f"  Transactions: {n_tx:,}")
    print(f"  Date range : {date_range}")
    print(f"  Treated    : {treated_pct:.1f}% of customers")
    print(f"\n  Hourly volume (threshold={thresh:.0f}):")
    for h, v in hourly.items():
        flag = " ← PEAK" if h in peak_detected else ""
        print(f"    {h:02d}:00  {v:5,}{flag}")
    print(f"\n  Peak hours detected: {peak_detected}")

    print(f"\n  OOS Product '{OOS_PRODUCT}':")
    n_days_total = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
    daily_rate = oos_total / n_days_total
    initial_est = round(daily_rate * 14 * 1.5)
    remaining   = max(0, initial_est - oos_last14)
    threshold_stock = max(5, initial_est * 0.20)
    in_stock    = remaining > threshold_stock
    print(f"    total_qty={oos_total}, daily_rate={daily_rate:.1f}")
    print(f"    initial_est={initial_est}, recent_demand(14d)={oos_last14}")
    print(f"    remaining={remaining}, threshold={threshold_stock:.0f}")
    print(f"    in_stock={in_stock}  ← {'✅ OOS triggered' if not in_stock else '⚠️  NOT triggered yet'}")

    # Bundle check
    bundled = df[df["product_name"].isin(BUNDLE_A + BUNDLE_B)]
    print(f"\n  Bundle products ({BUNDLE_A + BUNDLE_B}):")
    print(f"    transactions: {len(bundled):,} across {bundled['customer_id'].nunique()} customers")
    print(f"  {'='*58}")

    # ── Save transactions ────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH}")

    # ── Save separate labels file (for --promo-response) ────────────────────
    # aggregate per customer: max(is_treated), mean(post_promo_visit) → binary
    labels = (
        df.groupby("customer_id")
        .agg(
            is_treated      =("is_treated",       "max"),
            post_promo_visit=("post_promo_visit",  "mean"),
            promo_type      =("promo_type",        lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )
    median_ppv = labels["post_promo_visit"].median()
    labels["post_promo_visit"] = (labels["post_promo_visit"] > median_ppv).astype(int)
    labels.to_csv(LABELS_PATH, index=False)
    print(f"  Saved: {LABELS_PATH}")
    print(f"\n  Run pipeline:")
    print(f"    uv run --with 'pandas,numpy,lightgbm,scikit-learn,mlxtend' \\")
    print(f"      python3 src/csv_pipeline_engine.py \\")
    print(f"      {OUT_PATH} \\")
    print(f"      --promo-response {LABELS_PATH}")
    print(f"{'='*60}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(seed=args.seed)

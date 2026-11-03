"""
generate_full_feature_dataset.py — ML-valid synthetic retail dataset (demo / paper)

Rebuilt from scratch.  Previous version was designed for code-path coverage;
this version is designed for model validation.

Design goals
────────────
1. Heterogeneous τ(x) correlated with RFM
     → X-learner learns a real signal, not noise
2. Six latent customer types, no explicit labels in output
     → k-means must discover segments from behaviour, not pre-painted blobs
3. Bundle signal concentrated on hero products
     → Apriori association rules fire (support >> pipeline threshold)
4. Weekend (+40%) and month-end / payday (+20%) temporal lift
     → Layer 3 peak-hour detection reflects realistic traffic shape
5. Lapsing customers churn mid-window
     → Recency becomes genuinely informative for the model
6. OOS demand surge in last 14 days
     → Layer 3 out-of-stock guardrail triggers on real demand signal
7. 2% return transactions (negative quantity)
     → Layer 4 data-hygiene filter has real work to do

Outputs
───────
  data/synthetic_full_features.csv          transaction history (no labels)
  data/synthetic_full_features_labels.csv   per-customer treatment + outcome
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from math import exp

import numpy as np
import pandas as pd

# ── Timeline ───────────────────────────────────────────────────────────────────
START_DATE = date(2026, 7, 1)
END_DATE   = date(2026, 10, 29)   # 120-day observation window
N_DAYS     = (END_DATE - START_DATE).days  # 120

# ── Products ───────────────────────────────────────────────────────────────────
# Two hero bundle pairs — concentrated so Apriori support >> threshold
HERO_A, HERO_B = "Coffee-Mug",   "Tea-Infuser"    # bundle pair 1
HERO_C, HERO_D = "Notebook-A5",  "Ballpoint-Pen"  # bundle pair 2

# Out-of-stock product (demand surge injected in last 14 days)
OOS_PRODUCT = "Winter-Candle-Set"

# Long-tail (15 items)
LONG_TAIL = [
    "Coaster-Set",    "Wall-Art-Print", "Plant-Pot",      "Storage-Box",
    "Ceramic-Vase",   "Mirror-Frame",   "Door-Mat",       "Cushion-Cover",
    "Throw-Blanket",  "Photo-Frame",    "Desk-Organiser", "Scented-Candle",
    "Gift-Wrap-Kit",  "Cotton-Tote",    "Bamboo-Tray",
]

# Per-product unit_price ranges
PRICE: dict[str, tuple[float, float]] = {
    HERO_A:       (8.0,   15.0),
    HERO_B:       (12.0,  25.0),
    HERO_C:       (5.0,   12.0),
    HERO_D:       (3.0,    8.0),
    OOS_PRODUCT:  (25.0,  50.0),
    "Coaster-Set":    (15.0,  40.0),
    "Wall-Art-Print": (80.0, 250.0),
    "Plant-Pot":      (20.0,  60.0),
    "Storage-Box":    (25.0,  70.0),
    "Ceramic-Vase":   (40.0, 120.0),
    "Mirror-Frame":   (60.0, 180.0),
    "Door-Mat":       (20.0,  55.0),
    "Cushion-Cover":  (18.0,  45.0),
    "Throw-Blanket":  (30.0,  80.0),
    "Photo-Frame":    (15.0,  50.0),
    "Desk-Organiser": (20.0,  60.0),
    "Scented-Candle": (12.0,  35.0),
    "Gift-Wrap-Kit":  (8.0,   20.0),
    "Cotton-Tote":    (10.0,  30.0),
    "Bamboo-Tray":    (25.0,  70.0),
}

# ── Shopping hours ─────────────────────────────────────────────────────────────
PEAK_HOURS     = [18, 19, 20, 21]
OFF_PEAK_HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17]
ALL_HOURS      = OFF_PEAK_HOURS + PEAK_HOURS

# ── Latent customer types ──────────────────────────────────────────────────────
# Labels NOT in output — k-means must rediscover these from RFM.
#
# tau_base   : expected treatment effect for this type
# visits     : (min, max) visits in the full 120-day window
# aov        : (min, max) target order value per visit (THB / GBP / USD — unit-neutral)
# hours      : preferred shopping hours
# churn_day  : (min, max) day offset after which the customer goes silent; None = no churn
TYPES: dict[str, dict] = {
    "loyal": {
        "n": 90,
        "visits": (20, 40),
        "aov":    (180, 450),
        "tau_base": 0.08,      # already buying → small incremental uplift
        "hours": PEAK_HOURS,
        "churn_day": None,
    },
    "deal_seeker": {
        "n": 150,
        "visits": (8, 18),
        "aov":    (80, 200),
        "tau_base": 0.34,      # buys primarily on promo → largest uplift
        "hours": PEAK_HOURS,
        "churn_day": None,
    },
    "occasional": {
        "n": 90,
        "visits": (3, 8),
        "aov":    (280, 650),  # low frequency, large basket
        "tau_base": 0.17,
        "hours": OFF_PEAK_HOURS,
        "churn_day": None,
    },
    "lapsing": {
        "n": 120,
        "visits": (8, 18),     # active only in the early window
        "aov":    (100, 280),
        "tau_base": 0.26,      # winback opportunity → high uplift
        "hours": OFF_PEAK_HOURS,
        "churn_day": (25, 65), # go silent between day 25 and 65
    },
    "new_cust": {
        "n": 90,
        "visits": (2, 6),      # recently onboarded
        "aov":    (50, 150),
        "tau_base": 0.14,
        "hours": ALL_HOURS,
        "churn_day": None,
    },
    "low_value": {
        "n": 60,
        "visits": (2, 6),
        "aov":    (18, 55),    # low basket → triggers guardrail
        "tau_base": 0.04,      # near-zero treatment effect
        "hours": OFF_PEAK_HOURS,
        "churn_day": None,
    },
}
# Total: 90+150+90+120+90+60 = 600 customers


# ── Temporal weights ───────────────────────────────────────────────────────────
def _day_weight(d: date) -> float:
    w = 1.0
    if d.weekday() >= 5:   # Saturday / Sunday
        w *= 1.40
    if d.day >= 28:        # month-end payday spike
        w *= 1.20
    return w


def _date_weights(n_days: int) -> np.ndarray:
    days = [START_DATE + timedelta(days=i) for i in range(n_days)]
    w = np.array([_day_weight(d) for d in days], dtype=float)
    return w / w.sum()


# ── Basket generation ──────────────────────────────────────────────────────────
def _gen_basket(
    rng: np.random.Generator,
    ctype: str,
    aov_lo: float,
    aov_hi: float,
) -> list[tuple[str, float, int]]:
    """
    One shopping basket → list of (product, unit_price, qty).

    Hero bundles are co-purchased with high probability so that
    Apriori sees support >> pipeline threshold.
    """
    items: list[tuple[str, float, int]] = []

    # Loyal and deal-seekers are drawn to hero products more
    hero_boost = 1.45 if ctype in ("loyal", "deal_seeker") else 1.0

    # Hero pair 1: Coffee-Mug  + Tea-Infuser (72% co-purchase)
    if rng.random() < min(0.44 * hero_boost, 0.92):
        items.append((HERO_A, float(rng.uniform(*PRICE[HERO_A])), 1))
        if rng.random() < 0.72:
            items.append((HERO_B, float(rng.uniform(*PRICE[HERO_B])), 1))

    # Hero pair 2: Notebook-A5 + Ballpoint-Pen (66% co-purchase)
    if rng.random() < min(0.34 * hero_boost, 0.82):
        items.append((HERO_C, float(rng.uniform(*PRICE[HERO_C])), 1))
        if rng.random() < 0.66:
            items.append((HERO_D, float(rng.uniform(*PRICE[HERO_D])), 1))

    # Fill up to AOV target from long-tail
    target = float(rng.uniform(aov_lo, aov_hi))
    current = sum(p * q for _, p, q in items)
    fill_count = 0
    while current < target * 0.75 and fill_count < 6:
        prod  = str(rng.choice(LONG_TAIL))
        lo, hi = PRICE[prod]
        qty   = int(rng.integers(2, 5)) if rng.random() < 0.08 else 1
        price = float(rng.uniform(lo, hi))
        items.append((prod, price, qty))
        current += price * qty
        fill_count += 1

    # Guarantee at least one item
    if not items:
        prod = str(rng.choice(LONG_TAIL))
        items.append((prod, float(rng.uniform(*PRICE[prod])), 1))

    return items


# ── Treatment effect and conversion ───────────────────────────────────────────
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-float(np.clip(x, -20, 20))))


def _true_tau(
    ctype: str,
    freq_n: float,
    mon_n: float,
    rec_n: float,
    rng: np.random.Generator,
) -> float:
    """
    Heterogeneous treatment effect τ(x) — deliberately correlated with
    observable RFM so the X-learner has real signal to learn.

    Intuition
    ─────────
    High frequency  → lower τ  (loyal buyers don't need nudging)
    High recency    → higher τ  (lapsed customers respond to winback)
    Mid monetary    → highest τ  (deal-seekers sit in middle of spend range)
    """
    base = TYPES[ctype]["tau_base"]
    mod  = (
        -0.18 * freq_n
        + 0.12 * rec_n
        + 0.08 * (1.0 - abs(mon_n - 0.5) * 2.0)   # peak at mon_n = 0.5
    )
    noise = float(rng.normal(0, 0.02))
    return float(np.clip(base + mod + noise, 0.01, 0.70))


def _conv_prob(
    freq_n: float,
    mon_n: float,
    rec_n: float,
    treated: int,
    tau: float,
) -> float:
    """
    P(purchase after campaign) = sigmoid(log-odds from RFM) + T·τ(x)

    Loyal  (high freq, low rec)  → high base  ≈ 50%
    Lapsing (high rec)           → low base   ≈  8%  but large τ
    Low-value (low mon, low freq)→ low base   ≈  8%  and tiny τ
    """
    log_odds = -2.2 + 1.8 * freq_n + 0.9 * mon_n - 1.0 * rec_n
    base = _sigmoid(log_odds)
    return float(np.clip(base + treated * tau, 0.02, 0.95))


# ── Main generator ─────────────────────────────────────────────────────────────
def generate(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    day_idx  = np.arange(N_DAYS)
    dw       = _date_weights(N_DAYS)
    all_days = [START_DATE + timedelta(days=int(i)) for i in range(N_DAYS)]

    rows:   list[dict] = []
    c_meta: list[dict] = []   # (customer_id, ctype, churn_day)
    cid_n  = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1 — Transaction history
    # ──────────────────────────────────────────────────────────────────────────
    for ctype, cfg in TYPES.items():
        for _ in range(cfg["n"]):
            cid = f"C{cid_n:04d}"
            cid_n += 1

            n_visits = int(rng.integers(cfg["visits"][0], cfg["visits"][1] + 1))

            # Churn: lapsing customers go silent after churn_day
            if cfg["churn_day"] is not None:
                churn = int(rng.integers(cfg["churn_day"][0], cfg["churn_day"][1] + 1))
            else:
                churn = N_DAYS
            c_meta.append({"customer_id": cid, "ctype": ctype, "churn_day": churn})

            # Sample visit days with temporal weights, restricted to pre-churn window
            mask = day_idx < churn
            w_active = dw * mask
            if w_active.sum() == 0:
                continue
            w_active /= w_active.sum()
            visit_days = sorted(
                rng.choice(day_idx, size=n_visits, p=w_active).tolist()
            )

            for v_idx, day_off in enumerate(visit_days):
                vdate  = START_DATE + timedelta(days=int(day_off))
                hour   = int(rng.choice(cfg["hours"]))
                minute = int(rng.integers(0, 60))
                ts     = f"{vdate} {hour:02d}:{minute:02d}:00"
                inv_id = f"{cid}_{vdate}_{v_idx}"

                for prod, uprice, qty in _gen_basket(
                    rng, ctype, cfg["aov"][0], cfg["aov"][1]
                ):
                    rows.append({
                        "customer_id":  cid,
                        "timestamp":    ts,
                        "invoice_id":   inv_id,
                        "product_name": prod,
                        "unit_price":   round(uprice, 2),
                        "quantity":     qty,
                    })

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2 — OOS demand surge (last 14 days)
    # 60 "occasional" customers are OOS-affine; surge quantity >> initial estimate
    # ──────────────────────────────────────────────────────────────────────────
    oos_affine = [m["customer_id"] for m in c_meta if m["ctype"] == "occasional"][:60]
    oos_surge_start = END_DATE - timedelta(days=14)

    for cid in oos_affine:
        # Baseline OOS purchases before surge window
        for i in range(int(rng.integers(2, 5))):
            day_off = int(rng.integers(0, N_DAYS - 14))
            vdate   = START_DATE + timedelta(days=day_off)
            ts      = f"{vdate} {int(rng.choice(OFF_PEAK_HOURS)):02d}:{int(rng.integers(0,60)):02d}:00"
            rows.append({
                "customer_id":  cid,
                "timestamp":    ts,
                "invoice_id":   f"{cid}_oos_base_{i}",
                "product_name": OOS_PRODUCT,
                "unit_price":   round(float(rng.uniform(*PRICE[OOS_PRODUCT])), 2),
                "quantity":     int(rng.integers(1, 3)),
            })

        # Surge: high-quantity orders in last 14 days (drives recent_demand >> initial_est)
        for i in range(int(rng.integers(4, 9))):
            day_off = int(rng.integers(N_DAYS - 14, N_DAYS))
            vdate   = START_DATE + timedelta(days=day_off)
            ts      = f"{vdate} {int(rng.choice(OFF_PEAK_HOURS)):02d}:{int(rng.integers(0,60)):02d}:00"
            rows.append({
                "customer_id":  cid,
                "timestamp":    ts,
                "invoice_id":   f"{cid}_oos_surge_{i}",
                "product_name": OOS_PRODUCT,
                "unit_price":   round(float(rng.uniform(*PRICE[OOS_PRODUCT])), 2),
                "quantity":     int(rng.integers(8, 22)),   # high qty = demand signal
            })

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3 — Peak-hour background traffic
    # Adds 500 extra transactions in 18-21h from loyal/deal_seeker customers
    # so that hourly volume in 18-21 >> mean + 0.5·std  (threshold detection)
    # ──────────────────────────────────────────────────────────────────────────
    peak_customers = [m["customer_id"] for m in c_meta
                      if m["ctype"] in ("loyal", "deal_seeker")]
    for bg in range(500):
        cid     = str(rng.choice(peak_customers))
        day_off = int(rng.choice(day_idx, p=dw))
        vdate   = START_DATE + timedelta(days=day_off)
        hour    = int(rng.choice(PEAK_HOURS))
        minute  = int(rng.integers(0, 60))
        ts      = f"{vdate} {hour:02d}:{minute:02d}:00"
        prod    = str(rng.choice([HERO_A, HERO_B] + LONG_TAIL[:5]))
        rows.append({
            "customer_id":  cid,
            "timestamp":    ts,
            "invoice_id":   f"{cid}_pk_{bg}",
            "product_name": prod,
            "unit_price":   round(float(rng.uniform(*PRICE[prod])), 2),
            "quantity":     1,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4 — Returns (2% of rows, negative quantity)
    # Pipeline filters amount <= 0 so these test data-hygiene logic
    # ──────────────────────────────────────────────────────────────────────────
    n_base = len(rows)
    for idx in rng.choice(n_base, size=int(n_base * 0.02), replace=False):
        orig = rows[idx]
        if orig["product_name"] == OOS_PRODUCT:
            continue  # keep OOS demand signal clean
        ret = dict(orig)
        ret["quantity"]   = -ret["quantity"]
        ret["invoice_id"] = orig["invoice_id"] + "_RET"
        try:
            ret_ts = pd.Timestamp(orig["timestamp"]) + pd.Timedelta(
                days=int(rng.integers(1, 4))
            )
            ret["timestamp"] = str(ret_ts)
        except Exception:
            pass
        rows.append(ret)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5 — Build transaction DataFrame
    # ──────────────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("transaction_date").reset_index(drop=True)

    df_pos = df[df["quantity"] > 0].copy()
    df_pos["amount"] = df_pos["unit_price"] * df_pos["quantity"]

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 6 — Compute RFM and normalise (for τ modulation)
    # ──────────────────────────────────────────────────────────────────────────
    ref = df_pos["transaction_date"].max() + pd.Timedelta(days=1)
    rfm = df_pos.groupby("customer_id").agg(
        recency  =("transaction_date", lambda x: (ref - x.max()).days),
        frequency=("invoice_id", "nunique"),
        monetary =("amount", "sum"),
    ).reset_index()

    # Clip-normalise to [0,1] using 1st–99th percentile
    for col in ("recency", "frequency", "monetary"):
        lo_p = rfm[col].quantile(0.01)
        hi_p = rfm[col].quantile(0.99)
        rfm[f"{col}_n"] = ((rfm[col] - lo_p) / (hi_p - lo_p + 1e-9)).clip(0, 1)

    rfm_idx = rfm.set_index("customer_id")[
        ["recency_n", "frequency_n", "monetary_n"]
    ].to_dict("index")

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 7 — Treatment assignment + conversion (labels only — not in txn CSV)
    # ──────────────────────────────────────────────────────────────────────────
    meta_df = pd.DataFrame(c_meta)
    meta_df["is_treated"] = rng.binomial(1, 0.40, len(meta_df))

    label_rows = []
    for _, row in meta_df.iterrows():
        cid     = row["customer_id"]
        ctype   = row["ctype"]
        treated = int(row["is_treated"])
        norms   = rfm_idx.get(cid, {"recency_n": 0.5, "frequency_n": 0.5, "monetary_n": 0.5})

        tau    = _true_tau(ctype, norms["frequency_n"], norms["monetary_n"],
                           norms["recency_n"], rng)
        prob   = _conv_prob(norms["frequency_n"], norms["monetary_n"],
                            norms["recency_n"], treated, tau)
        bought = int(rng.binomial(1, prob))

        aov_lo, aov_hi = TYPES[ctype]["aov"]
        order_value = round(float(rng.uniform(aov_lo, aov_hi)), 2) if bought else 0.0

        label_rows.append({
            "customer_id":      cid,
            "is_treated":       treated,
            "post_promo_visit": bought,
            "order_value":      order_value,
        })

    labels_df = pd.DataFrame(label_rows)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 8 — Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    # Peak-hour detection
    df_pos["hour"] = df_pos["transaction_date"].dt.hour
    hourly    = df_pos.groupby("hour").size()
    h_mean    = hourly.mean()
    h_std     = hourly.std()
    threshold = h_mean + 0.5 * h_std
    peak_det  = sorted([h for h, v in hourly.items() if v > threshold])

    # OOS check
    oos_txns  = df_pos[df_pos["product_name"] == OOS_PRODUCT]
    daily_r   = len(oos_txns) / N_DAYS
    init_est  = int(daily_r * 60)
    surge_cut = pd.Timestamp(oos_surge_start)
    rec_dem   = int(oos_txns[oos_txns["transaction_date"] >= surge_cut]["quantity"].sum())
    remaining = max(0, init_est - rec_dem)
    in_stock  = remaining > max(5, int(init_est * 0.20))

    # Bundle support for Apriori validation
    n_inv = df_pos["invoice_id"].nunique()
    ab_inv = df_pos[df_pos["product_name"].isin([HERO_A, HERO_B])].groupby(
        "invoice_id")["product_name"].nunique()
    ab_sup = (ab_inv >= 2).sum() / max(n_inv, 1)
    cd_inv = df_pos[df_pos["product_name"].isin([HERO_C, HERO_D])].groupby(
        "invoice_id")["product_name"].nunique()
    cd_sup = (cd_inv >= 2).sum() / max(n_inv, 1)
    min_sup = 0.05 if n_inv > 5000 else (0.02 if n_inv > 1000 else 0.01)

    # Conversion by type (shows τ heterogeneity)
    # meta_df already has is_treated; merge only the outcome column from labels_df
    stats = meta_df.merge(
        labels_df[["customer_id", "post_promo_visit"]],
        on="customer_id",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 9 — Export
    # ──────────────────────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    txn_cols = ["customer_id", "timestamp", "invoice_id",
                "product_name", "unit_price", "quantity"]
    df[txn_cols].to_csv("data/synthetic_full_features.csv", index=False)
    labels_df.to_csv("data/synthetic_full_features_labels.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic report
    # ──────────────────────────────────────────────────────────────────────────
    n_cust = df["customer_id"].nunique()
    n_tx   = len(df)
    tr_n   = labels_df["is_treated"].sum()
    cv_t   = labels_df[labels_df["is_treated"] == 1]["post_promo_visit"].mean()
    cv_c   = labels_df[labels_df["is_treated"] == 0]["post_promo_visit"].mean()

    print("=" * 62)
    print("  ML-Valid Synthetic Dataset (demo / paper)")
    print("=" * 62)
    print(f"  Customers    : {n_cust:,}")
    print(f"  Transactions : {n_tx:,}  (incl. {int(n_tx/n_cust)} avg / customer)")
    print(f"  Baskets      : {n_inv:,}  ({n_inv/n_cust:.1f} avg / customer)")
    print(f"  Date range   : {START_DATE} → {df['transaction_date'].max().date()}")
    print(f"  Treated      : {tr_n} ({tr_n/len(labels_df)*100:.0f}%)")
    print()
    print(f"  Overall conversion:")
    print(f"    Treated  : {cv_t:.1%}")
    print(f"    Control  : {cv_c:.1%}")
    print(f"    True lift: {cv_t - cv_c:+.1%}  ← X-learner should recover this")
    print()
    print("  τ(x) heterogeneity by latent type  (validates correlated uplift):")
    print(f"  {'type':12s}  {'n':>4}  {'treated':>8}  {'control':>8}  {'lift':>7}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*7}")
    for ctype in TYPES:
        sub = stats[stats["ctype"] == ctype]
        t   = sub[sub["is_treated"] == 1]["post_promo_visit"].mean()
        c   = sub[sub["is_treated"] == 0]["post_promo_visit"].mean()
        lft = t - c if not (pd.isna(t) or pd.isna(c)) else float("nan")
        print(f"  {ctype:12s}  {len(sub):>4}  {t:>8.1%}  {c:>8.1%}  {lft:>+7.1%}")
    print()

    print(f"  Hourly volume (threshold={threshold:.0f}):")
    for h, v in sorted(hourly.items()):
        flag = "  ← PEAK" if h in peak_det else ""
        print(f"    {h:02d}:00  {int(v):5d}{flag}")
    print(f"  Peak hours detected: {peak_det}")
    print()

    print(f"  OOS product '{OOS_PRODUCT}':")
    print(f"    initial_est={init_est}, recent_demand(14d)={rec_dem}")
    print(f"    remaining={remaining}  →  in_stock={in_stock}",
          "← ✅ OOS triggered" if not in_stock else "← ❌ NOT triggered")
    print()

    print(f"  Apriori bundle support (min_support={min_sup}):")
    print(f"    {HERO_A}+{HERO_B}: {ab_sup:.3f}",
          "✅" if ab_sup > min_sup else "❌")
    print(f"    {HERO_C}+{HERO_D}: {cd_sup:.3f}",
          "✅" if cd_sup > min_sup else "❌")
    print()

    print("  Saved: data/synthetic_full_features.csv")
    print("  Saved: data/synthetic_full_features_labels.csv")
    print()
    print("  Run pipeline:")
    print("    uv run --with 'pandas,numpy,lightgbm,scikit-learn,mlxtend' \\")
    print("      python3 src/csv_pipeline_engine.py \\")
    print("      data/synthetic_full_features.csv \\")
    print("      --promo-response data/synthetic_full_features_labels.csv")
    print("=" * 62)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ML-valid synthetic retail dataset for demo / paper"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    generate(seed=args.seed)

"""
convert_starbucks.py — แปลง Starbucks Promotional Offer Dataset → POS format ที่โมเดลรับได้

Input files (download จาก Kaggle: "Starbucks Capstone Challenge"):
  data/starbucks/portfolio.json   — offer metadata
  data/starbucks/profile.json     — customer demographics
  data/starbucks/transcript.json  — event log (transactions + offer events)

Output:
  data/starbucks/starbucks_pos.csv    — POS transactions (ใช้กับ run_layer4 แบบ proxy label)
  data/starbucks/starbucks_labels.csv — real A/B labels (is_treated, post_promo_visit, promo_type)

Usage:
  python scripts/convert_starbucks.py
  python scripts/convert_starbucks.py --input-dir path/to/starbucks --output-dir data/starbucks

จากนั้น serve:
  python src/train_champion.py \\
      --store-data "starbucks:data/starbucks/starbucks_pos.csv:data/starbucks/starbucks_labels.csv" \\
      --baseline-data data/starbucks/starbucks_pos.csv \\
      --models-root models

หรือดู results:
  python -c "
  import sys; sys.path.insert(0, 'src')
  import campaign_pipeline as cp
  out = cp.run_layer4('data/starbucks/starbucks_pos.csv',
                      'data/starbucks/starbucks_labels.csv',
                      models_root='models')
  print(out[['customer_id','recommended_action','expected_profit','should_send']]
        .sort_values('expected_profit', ascending=False).head(20))
  "

Dataset notes:
  - ข้อมูลเป็น SIMULATED (ไม่ใช่ real Starbucks) แต่ pattern realistic
  - time column = hours from test start, ref date = 2018-01-01
  - ไม่มี product_name/quantity จริง → สร้าง synthetic ใกล้เคียงจาก amount
  - Treatment = ลูกค้าได้รับ discount/BOGO offer อย่างน้อย 1 ครั้ง
    (informational offer ไม่นับเพราะไม่มี incentive)
  - post_promo_visit = ลูกค้า treated ที่ complete offer อย่างน้อย 1 ครั้ง
    สำหรับ control = ซื้อในช่วง 2nd half ของ timeline
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Coffee shop product catalogue (synthesized from spend amount) ──────────────
# Price tiers match typical Starbucks transaction sizes
_PRODUCTS = [
    ("Espresso",            2.45),
    ("Drip Coffee",         2.25),
    ("Latte",               4.45),
    ("Cappuccino",          4.15),
    ("Americano",           3.45),
    ("Cold Brew",           4.75),
    ("Frappuccino",         5.25),
    ("Matcha Latte",        5.45),
    ("Hot Chocolate",       3.95),
    ("Chai Tea Latte",      4.75),
    ("Iced Tea",            3.25),
    ("Breakfast Sandwich",  4.95),
    ("Cake Pop",            2.45),
    ("Cookie",              1.95),
]
_PROD_NAMES  = [p[0] for p in _PRODUCTS]
_PROD_PRICES = np.array([p[1] for p in _PRODUCTS])

REF_DATE = pd.Timestamp("2018-01-01")   # anchor for time-in-hours conversion


def _assign_products(amount: float, rng: np.random.Generator) -> list[dict]:
    """
    Decompose a transaction amount into 1-3 line items.
    Picks products whose prices sum closest to `amount`.
    """
    if amount <= 0:
        return []
    items = []
    remaining = amount
    n_items = int(rng.integers(1, 4))  # 1-3 items per visit
    for _ in range(n_items):
        if remaining < 1.5:
            break
        # Pick product whose price ≤ remaining, weighted toward cheaper items
        eligible = _PROD_PRICES <= remaining + 0.5
        if not eligible.any():
            break
        weights = 1.0 / (_PROD_PRICES[eligible] + 0.01)
        weights /= weights.sum()
        idx = int(rng.choice(np.where(eligible)[0], p=weights))
        qty = max(1, int(round(remaining / _PROD_PRICES[idx])))
        qty = min(qty, 4)  # cap at 4 (no bulk coffee orders)
        price = round(_PROD_PRICES[idx] * qty, 2)
        items.append({
            "product_name": _PROD_NAMES[idx],
            "quantity":     qty,
            "total_price":  price,
        })
        remaining -= price
        if remaining < 1.0:
            break
    # Guarantee at least one item with the full amount
    if not items:
        idx = int(np.argmin(np.abs(_PROD_PRICES - amount)))
        items.append({
            "product_name": _PROD_NAMES[idx],
            "quantity":     1,
            "total_price":  round(amount, 2),
        })
    return items


def load_starbucks(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three Starbucks JSON files."""
    def _read(name):
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            sys.exit(f"❌ File not found: {path}\n"
                     f"   Download from Kaggle: 'Starbucks Capstone Challenge'")
        return pd.read_json(path, orient="records", lines=False)

    portfolio = _read("portfolio.json")
    profile   = _read("profile.json")
    transcript = _read("transcript.json")
    return portfolio, profile, transcript


def build_offer_map(portfolio: pd.DataFrame) -> dict:
    """offer_id → {type, duration_hours, reward, difficulty}"""
    omap = {}
    for _, row in portfolio.iterrows():
        omap[row["id"]] = {
            "offer_type":      row["offer_type"],          # bogo / discount / informational
            "duration_hours":  int(row["duration"]) * 24,
            "reward":          row.get("reward", 0),
            "difficulty":      row.get("difficulty", 0),
        }
    return omap


def expand_transcript(transcript: pd.DataFrame, offer_map: dict) -> pd.DataFrame:
    """Flatten the value dict column and tag each row with event details."""
    rows = []
    for _, r in transcript.iterrows():
        val = r["value"] if isinstance(r["value"], dict) else {}
        offer_id = val.get("offer id") or val.get("offer_id")
        amount   = val.get("amount", 0.0)
        rows.append({
            "customer_id": str(r["person"]),
            "event":       r["event"],
            "time_h":      float(r["time"]),
            "offer_id":    offer_id,
            "amount":      float(amount),
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = REF_DATE + pd.to_timedelta(df["time_h"], unit="h")

    # Tag offer metadata
    df["offer_type"] = df["offer_id"].map(
        lambda oid: offer_map.get(oid, {}).get("offer_type") if pd.notna(oid) else None
    )
    df["offer_duration_h"] = df["offer_id"].map(
        lambda oid: offer_map.get(oid, {}).get("duration_hours") if pd.notna(oid) else None
    )
    return df


def build_labels(events: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer A/B labels.

    Treatment definition:
      is_treated = 1  →  received at least one discount or BOGO offer
                         (informational excluded: no purchase incentive)

    Conversion definition (post_promo_visit):
      Treated:  completed at least one discount/BOGO offer
      Control:  made a purchase in the 2nd half of the observation window
                (time_h > median(max_time))  — temporal proxy for retention
    """
    # ── Treated customers ────────────────────────────────────────────────────
    incentive_offers = events[
        (events["event"] == "offer received") &
        (events["offer_type"].isin(["bogo", "discount"]))
    ]
    treated_ids = set(incentive_offers["customer_id"])

    # ── Offer completions (treated response) ─────────────────────────────────
    completions = events[
        (events["event"] == "offer completed") &
        (events["offer_type"].isin(["bogo", "discount"]))
    ]
    completed_ids = set(completions["customer_id"])

    # ── Control response: purchase in 2nd half of timeline ───────────────────
    max_time = events["time_h"].max()
    late_txn = events[
        (events["event"] == "transaction") &
        (events["time_h"] > max_time / 2)
    ]
    late_buyers = set(late_txn["customer_id"])

    # ── Promo type for treated customers (most frequent offer type received) ──
    promo_type_map = (
        incentive_offers.groupby("customer_id")["offer_type"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    all_customers = events["customer_id"].unique()
    records = []
    for cid in all_customers:
        treated = cid in treated_ids
        if treated:
            converted = int(cid in completed_ids)
            ptype = promo_type_map.get(cid, "discount")
        else:
            converted = int(cid in late_buyers)
            ptype = "none"
        records.append({
            "customer_id":      cid,
            "is_treated":       int(treated),
            "post_promo_visit": converted,
            "promo_type":       ptype,
        })
    return pd.DataFrame(records)


def build_pos(events: pd.DataFrame, labels: pd.DataFrame,
              rng: np.random.Generator) -> pd.DataFrame:
    """
    Transaction events → POS line-item rows.
    Each Starbucks transaction (no SKU) is decomposed into 1-3 synthetic line items.
    """
    txn_events = events[events["event"] == "transaction"].copy()
    txn_events = txn_events[txn_events["amount"] > 0]

    label_lookup = labels.set_index("customer_id")[
        ["is_treated", "post_promo_visit", "promo_type"]
    ].to_dict("index")

    rows = []
    txn_counter = 0
    for _, row in txn_events.iterrows():
        cid  = row["customer_id"]
        ts   = row["timestamp"]
        amt  = row["amount"]
        lab  = label_lookup.get(cid, {})

        items = _assign_products(amt, rng)
        basket_id = f"SB{txn_counter:08d}"
        txn_counter += 1

        for item in items:
            rows.append({
                "transaction_id":  basket_id,
                "customer_id":     cid,
                "timestamp":       ts.strftime("%Y-%m-%d %H:%M:%S"),
                "product_name":    item["product_name"],
                "quantity":        item["quantity"],
                "total_price":     item["total_price"],
                # Real A/B labels (written into POS so labels file = same file or separate)
                "is_treated":      lab.get("is_treated", 0),
                "post_promo_visit":lab.get("post_promo_visit", 0),
                "promo_type":      lab.get("promo_type", "none"),
            })

    pos = pd.DataFrame(rows)
    pos = pos[pos["total_price"] > 0]
    return pos


def print_summary(pos: pd.DataFrame, labels: pd.DataFrame) -> None:
    n_cust    = pos["customer_id"].nunique()
    n_txn     = pos["transaction_id"].nunique()
    n_rows    = len(pos)
    treat     = labels["is_treated"].mean()
    conv_t    = labels[labels["is_treated"] == 1]["post_promo_visit"].mean()
    conv_c    = labels[labels["is_treated"] == 0]["post_promo_visit"].mean()
    uplift    = conv_t - conv_c
    date_min  = pd.to_datetime(pos["timestamp"]).min().date()
    date_max  = pd.to_datetime(pos["timestamp"]).max().date()

    print(f"\n📊 Starbucks Conversion Summary")
    print(f"   Customers    : {n_cust:,}")
    print(f"   Baskets      : {n_txn:,}")
    print(f"   Line items   : {n_rows:,}")
    print(f"   Date range   : {date_min} → {date_max}")
    print(f"   Treated      : {treat:.1%}  (received discount/BOGO)")
    print(f"   Conv treated : {conv_t:.1%}  (completed offer)")
    print(f"   Conv control : {conv_c:.1%}  (late-period purchase)")
    print(f"   Raw uplift   : {uplift:+.1%}  (treat − control)")
    print(f"\n   → ถ้า raw uplift > 5% และ n_treat/n_control ไม่ต่างกันมาก dataset นี้ใช้ได้ดี")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",  default="data/starbucks",
                   help="โฟลเดอร์ที่เก็บ portfolio/profile/transcript.json")
    p.add_argument("--output-dir", default="data/starbucks",
                   help="โฟลเดอร์ที่จะบันทึกไฟล์ output")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed สำหรับ product assignment")
    p.add_argument("--no-labels-in-pos", action="store_true",
                   help="ไม่เขียน is_treated/post_promo_visit ลงใน POS file "
                        "(สร้างเป็นไฟล์ labels แยกต่างหากอย่างเดียว)")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("📂 Loading Starbucks files...")
    portfolio, profile, transcript = load_starbucks(args.input_dir)
    print(f"   portfolio: {len(portfolio)} offers | "
          f"profile: {len(profile):,} customers | "
          f"transcript: {len(transcript):,} events")

    offer_map = build_offer_map(portfolio)
    events    = expand_transcript(transcript, offer_map)

    print("🏷️  Building A/B labels...")
    labels = build_labels(events)

    print("🧾 Converting transactions to POS line items...")
    pos = build_pos(events, labels, rng)

    if args.no_labels_in_pos:
        # Strip label columns from POS (serve with separate labels file)
        pos_out = pos.drop(columns=["is_treated", "post_promo_visit", "promo_type"],
                           errors="ignore")
    else:
        pos_out = pos

    pos_path    = os.path.join(args.output_dir, "starbucks_pos.csv")
    labels_path = os.path.join(args.output_dir, "starbucks_labels.csv")

    pos_out.to_csv(pos_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"\n✅ Output written:")
    print(f"   POS    → {pos_path}  ({len(pos_out):,} rows)")
    print(f"   Labels → {labels_path}  ({len(labels):,} customers)")

    print_summary(pos_out, labels)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next steps:

1. Train a new challenger on Starbucks data:
   python src/train_champion.py \\
       --store-data "starbucks:{pos_path}:{labels_path}" \\
       --baseline-data {pos_path} \\
       --models-root models

2. Or serve with current champion (zero-shot test):
   python -c "
   import sys; sys.path.insert(0, 'src')
   import campaign_pipeline as cp
   out = cp.run_layer4('{pos_path}', '{labels_path}', models_root='models')
   send = out[out['should_send'].astype(bool)]
   print(f'Send: {{len(send)}}/{{len(out)}} ({{len(send)/len(out):.1%}})')
   print(send[['customer_id','recommended_action','expected_profit']].head(10))
   "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")


if __name__ == "__main__":
    main()

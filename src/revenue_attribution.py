"""revenue_attribution.py — Campaign revenue attribution and 5% share calculation.

Tracks which revenue came through Codera's channels so the 5% revenue-share fee
can be calculated fairly each month.

Attribution methods (applied in priority order)
------------------------------------------------
1. campaign_tag   — POS row has a ``campaign_id`` or ``source`` column
                    matching a known campaign_id in campaign_history.jsonl
2. customer_match — customer_id bought within attribution_window_days after
                    receiving a campaign (broadcast to that segment)
3. promo_code     — POS row has a ``promo_code`` that was issued by Codera

Only revenue in the attribution window counts. Transactions outside the window
or from customers who never received a campaign are marked as ``organic``.

Storage
-------
  data/attribution_log.jsonl  — one record per attributed transaction (append-only)

Usage
-----
    from revenue_attribution import compute_attributed_revenue, get_attribution_report
    import pandas as pd

    pos_df  = pd.read_csv("pos.csv")   # columns: customer_id, transaction_id, amount, date, [source, promo_code]
    history = load_campaign_history("data/campaign_history.jsonl")

    result = compute_attributed_revenue(
        pos_df, history, data_dir="data",
        revenue_share_pct=0.05,
    )
    print(result["attributed_revenue"])   # ยอดที่มาจากช่องทางของ Codera
    print(result["revenue_share_fee"])    # 5% ของยอดนั้น
    print(get_attribution_report(result)) # summary string ภาษาไทย
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta

_logger = logging.getLogger(__name__)

_DEFAULT_ATTRIBUTION_WINDOW_DAYS = 30   # ลูกค้าซื้อภายใน 30 วันหลังได้รับ campaign → attributed
_DEFAULT_REVENUE_SHARE_PCT       = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_attributed_revenue(
    pos_df,
    campaign_history: list[dict],
    data_dir: str = "data",
    attribution_window_days: int = _DEFAULT_ATTRIBUTION_WINDOW_DAYS,
    revenue_share_pct: float = _DEFAULT_REVENUE_SHARE_PCT,
    month: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Compute which revenue in pos_df is attributable to Codera campaigns.

    Parameters
    ----------
    pos_df : pd.DataFrame
        POS transaction data. Required columns: ``customer_id``, ``amount``, ``date``.
        Optional columns: ``campaign_id``, ``source``, ``promo_code``, ``transaction_id``.
    campaign_history : list[dict]
        Records from campaign_history.jsonl (executed campaigns only).
    data_dir : str
        Root data dir — attribution_log.jsonl is written here.
    attribution_window_days : int
        Days after campaign execution a purchase is attributed to that campaign.
    revenue_share_pct : float
        Revenue share rate (default 0.05 = 5%).
    month : str, optional
        Filter POS data to this month "YYYY-MM". None = all data.
    dry_run : bool
        If True, skip writing to attribution_log.jsonl.

    Returns
    -------
    dict
        ``{total_revenue, attributed_revenue, organic_revenue,
           attribution_rate, revenue_share_fee, attributed_transactions,
           by_method, by_campaign, month}``
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required")

    df = pos_df.copy()

    # Normalise date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "transaction_date" in df.columns:
        df["date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    else:
        _logger.warning("compute_attributed_revenue: no date column found")
        df["date"] = pd.NaT

    # Filter to month if requested
    if month:
        try:
            yr, mo = int(month[:4]), int(month[5:7])
            df = df[
                (df["date"].dt.year  == yr) &
                (df["date"].dt.month == mo)
            ]
        except Exception as exc:
            _logger.warning("compute_attributed_revenue: invalid month filter %s — %s", month, exc)

    df = df.dropna(subset=["date"])

    total_revenue = float(df["amount"].sum()) if "amount" in df.columns else 0.0

    # Build lookup structures from campaign history
    executed = [
        c for c in campaign_history
        if c.get("status", "").startswith("executed") or c.get("executed_at")
    ]

    campaign_id_set = {c.get("execution_id", "") for c in executed if c.get("execution_id")}
    campaign_id_set |= {c.get("plan_id", "")     for c in executed if c.get("plan_id")}
    campaign_id_set.discard("")

    # customer_id → list of (executed_at datetime, campaign_id)
    customer_campaign_map: dict[str, list[tuple[datetime, str]]] = {}
    for c in executed:
        exec_at_str = c.get("executed_at", "")
        if not exec_at_str:
            continue
        try:
            exec_at = datetime.fromisoformat(exec_at_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        # audience is stored per-plan — if delivery_stats has user_ids use them,
        # otherwise use segment targeting (best-effort)
        for uid in c.get("audience_customer_ids", []):
            customer_campaign_map.setdefault(uid, []).append(
                (exec_at, c.get("execution_id") or c.get("plan_id", ""))
            )

    # Tag each row
    attributed_rows: list[dict] = []
    methods: dict[str, int]      = {"campaign_tag": 0, "customer_match": 0, "promo_code": 0}
    by_campaign: dict[str, float] = {}

    for _, row in df.iterrows():
        method    = _attribute_row(row, campaign_id_set, customer_campaign_map,
                                   attribution_window_days)
        amount    = float(row.get("amount", 0) or 0)
        campaign  = str(row.get("campaign_id", "") or row.get("source", "") or "")

        if method:
            methods[method] = methods.get(method, 0) + 1
            by_campaign[campaign or "untagged"] = (
                by_campaign.get(campaign or "untagged", 0.0) + amount
            )
            attributed_rows.append({
                "transaction_id": str(row.get("transaction_id", "") or ""),
                "customer_id":    str(row.get("customer_id", "") or ""),
                "amount":         amount,
                "date":           str(row["date"].date()),
                "method":         method,
                "campaign_id":    campaign,
                "attributed_at":  datetime.now(timezone.utc).isoformat(),
                "month":          month or "",
            })

    attributed_revenue = sum(r["amount"] for r in attributed_rows)
    organic_revenue    = total_revenue - attributed_revenue
    attribution_rate   = round(attributed_revenue / total_revenue, 4) if total_revenue > 0 else 0.0
    revenue_share_fee  = round(attributed_revenue * revenue_share_pct, 2)

    if not dry_run and attributed_rows:
        _write_attribution_log(data_dir, attributed_rows)

    _logger.info(
        "compute_attributed_revenue: total=%.2f attributed=%.2f (%.1f%%) fee=%.2f",
        total_revenue, attributed_revenue, attribution_rate * 100, revenue_share_fee,
    )

    return {
        "total_revenue":           round(total_revenue, 2),
        "attributed_revenue":      round(attributed_revenue, 2),
        "organic_revenue":         round(organic_revenue, 2),
        "attribution_rate":        attribution_rate,
        "revenue_share_fee":       revenue_share_fee,
        "revenue_share_pct":       revenue_share_pct,
        "attributed_transactions": len(attributed_rows),
        "by_method":               methods,
        "by_campaign":             {k: round(v, 2) for k, v in by_campaign.items()},
        "month":                   month or "",
    }


def get_attribution_report(result: dict) -> str:
    """Return a one-page Thai summary suitable for the monthly billing report."""
    month_label = f" ({result['month']})" if result.get("month") else ""
    rate_pct    = result.get("attribution_rate", 0) * 100
    share_pct   = result.get("revenue_share_pct", 0.05) * 100

    lines = [
        f"รายงาน Revenue Attribution{month_label}",
        f"  ยอดขายรวม              : ฿{result['total_revenue']:,.2f}",
        f"  ยอดจากช่องทาง Codera  : ฿{result['attributed_revenue']:,.2f}  ({rate_pct:.1f}%)",
        f"  ยอด Organic            : ฿{result['organic_revenue']:,.2f}",
        f"  ค่าบริการ {share_pct:.0f}%          : ฿{result['revenue_share_fee']:,.2f}",
        f"  จำนวน transactions     : {result['attributed_transactions']:,} รายการ",
        "",
        "  วิธีระบุที่มา:",
    ]

    for method, count in result.get("by_method", {}).items():
        label = {
            "campaign_tag":    "Campaign tag ใน POS",
            "customer_match":  "Customer match (ซื้อใน window)",
            "promo_code":      "Promo code",
        }.get(method, method)
        lines.append(f"    {label:30}: {count:,} รายการ")

    if result.get("by_campaign"):
        lines.append("")
        lines.append("  ยอดแยกตาม campaign:")
        for cid, rev in sorted(result["by_campaign"].items(),
                                key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"    {cid:30}: ฿{rev:,.2f}")

    return "\n".join(lines)


def load_campaign_history(path: str) -> list[dict]:
    """Load campaign_history.jsonl from disk."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def tag_transaction_source(
    pos_df,
    campaign_id: str,
    customer_ids: list[str],
) -> any:
    """Add ``campaign_id`` column to POS rows belonging to campaign customers.

    Utility for stores that don't have a ``source`` column in their POS —
    call this after each broadcast to mark who received which campaign.

    Parameters
    ----------
    pos_df : pd.DataFrame
    campaign_id : str
    customer_ids : list[str]
        Customers who received this campaign.

    Returns
    -------
    pd.DataFrame
        Copy of pos_df with ``campaign_id`` column added/updated.
    """
    df = pos_df.copy()
    if "campaign_id" not in df.columns:
        df["campaign_id"] = ""
    mask = df["customer_id"].isin(set(customer_ids))
    df.loc[mask, "campaign_id"] = campaign_id
    _logger.info(
        "tag_transaction_source: tagged %d rows with campaign_id=%s",
        mask.sum(), campaign_id,
    )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _attribute_row(
    row,
    campaign_id_set: set[str],
    customer_campaign_map: dict,
    window_days: int,
) -> str | None:
    """Return attribution method string or None if organic."""
    # Method 1: explicit campaign_id or source column
    for col in ("campaign_id", "source"):
        val = str(row.get(col, "") or "").strip()
        if val and val in campaign_id_set:
            return "campaign_tag"

    # Method 2: customer received campaign within window
    cid = str(row.get("customer_id", "") or "").strip()
    row_date = row.get("date")
    if cid and cid in customer_campaign_map and row_date is not None:
        try:
            import pandas as pd
            if pd.notna(row_date):
                row_dt = row_date
                if hasattr(row_dt, "to_pydatetime"):
                    row_dt = row_dt.to_pydatetime()
                if row_dt.tzinfo is None:
                    row_dt = row_dt.replace(tzinfo=timezone.utc)
                for exec_at, _ in customer_campaign_map[cid]:
                    delta = (row_dt - exec_at).total_seconds() / 86400
                    if 0 <= delta <= window_days:
                        return "customer_match"
        except Exception:
            pass

    # Method 3: promo_code column
    promo = str(row.get("promo_code", "") or "").strip()
    if promo and promo.upper().startswith("CRM"):
        return "promo_code"

    return None


def _write_attribution_log(data_dir: str, rows: list[dict]) -> None:
    path = os.path.join(data_dir, "attribution_log.jsonl")
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

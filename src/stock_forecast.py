"""stock_forecast.py — Weekly demand forecast per item from anonymous transaction data.

Uses a 7-day rolling average baseline adjusted by day-of-week multipliers.
No customer identity required — works purely on item sales history.

Usage
-----
    from ingest import ingest_meripos
    from stock_forecast import forecast_demand, get_stock_alerts

    tx_df, item_df = ingest_meripos("data/transactions.csv")
    forecast = forecast_demand(item_df, weeks_ahead=4)
    alerts   = get_stock_alerts(forecast, item_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from business_config import CONFIG

_EMPTY_FORECAST = pd.DataFrame(
    columns=["item_name", "week_start", "predicted_qty", "lower_bound", "upper_bound"]
)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def forecast_demand(
    item_df: pd.DataFrame,
    weeks_ahead: int | None = None,
    min_history_days: int | None = None,
) -> pd.DataFrame:
    """Forecast weekly demand per item using rolling-average + day-of-week multipliers.

    Parameters
    ----------
    item_df          : item-level DataFrame (columns: item_name, qty, timestamp).
                       Produced by ``ingest.parse_meripos_items()``.
    weeks_ahead      : weeks to forecast (default: CONFIG["stock_forecast_weeks"]).
    min_history_days : minimum history required to forecast an item
                       (default: CONFIG["stock_min_history_days"]).

    Returns
    -------
    DataFrame — columns: item_name, week_start (ISO date str),
                predicted_qty, lower_bound, upper_bound
    """
    weeks_ahead      = weeks_ahead      or CONFIG.get("stock_forecast_weeks",   4)
    min_history_days = min_history_days or CONFIG.get("stock_min_history_days", 14)

    if item_df.empty or "item_name" not in item_df.columns or "timestamp" not in item_df.columns:
        return _EMPTY_FORECAST.copy()

    df = item_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["qty"]       = pd.to_numeric(df.get("qty", 1), errors="coerce").fillna(1.0)
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return _EMPTY_FORECAST.copy()

    # Day-of-week multipliers computed across all items
    df["dow"] = df["timestamp"].dt.dayofweek   # 0 = Mon
    dow_totals = df.groupby("dow")["qty"].sum()
    global_mean = dow_totals.mean()
    dow_weight: dict[int, float] = (
        {int(d): float(v / global_mean) for d, v in dow_totals.items()}
        if global_mean > 0
        else {d: 1.0 for d in range(7)}
    )

    max_date = df["timestamp"].max()
    forecast_start = (max_date + pd.Timedelta(days=1)).normalize()

    rows: list[dict] = []
    for item_name, group in df.groupby("item_name"):
        history_days = (group["timestamp"].max() - group["timestamp"].min()).days
        if history_days < min_history_days:
            continue

        # Daily totals
        daily = (
            group.assign(date=group["timestamp"].dt.normalize())
            .groupby("date")["qty"].sum()
            .sort_index()
        )

        # 7-day rolling mean & std
        rolling = daily.rolling(7, min_periods=3)
        baseline = float(rolling.mean().iloc[-1]) if not daily.empty else float(daily.mean())
        last_std  = rolling.std().iloc[-1]
        std = (
            float(last_std)
            if (not np.isnan(float(last_std)) and float(last_std) > 0)
            else max(float(daily.std(ddof=0)), baseline * 0.20)
        )

        for w in range(weeks_ahead):
            week_start = forecast_start + pd.Timedelta(weeks=w)
            weekly_pred = sum(
                baseline * dow_weight.get((week_start + pd.Timedelta(days=d)).dayofweek, 1.0)
                for d in range(7)
            )
            weekly_std = std * (7 ** 0.5)   # scale daily std → weekly
            rows.append({
                "item_name":    item_name,
                "week_start":   week_start.date().isoformat(),
                "predicted_qty": round(max(0.0, weekly_pred), 1),
                "lower_bound":  round(max(0.0, weekly_pred - 1.5 * weekly_std), 1),
                "upper_bound":  round(weekly_pred + 1.5 * weekly_std, 1),
            })

    return pd.DataFrame(rows) if rows else _EMPTY_FORECAST.copy()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def get_stock_alerts(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> list[dict]:
    """Generate actionable stock alerts by comparing forecast vs recent history.

    Parameters
    ----------
    forecast_df : output of ``forecast_demand()``.
    history_df  : item_df covering the recent historical window (last 4+ weeks).

    Returns
    -------
    list of dicts: {item_name, alert_type, severity, message_th}
    Sorted by severity: critical → warning → info.
    """
    spike_thresh = CONFIG.get("stock_spike_threshold", 0.30)
    slow_thresh  = CONFIG.get("stock_slow_threshold",  0.50)
    slow_weeks   = int(CONFIG.get("stock_slow_weeks",  3))

    alerts: list[dict] = []

    if forecast_df.empty or history_df.empty:
        return alerts

    # Historical weekly averages
    hist = history_df.copy()
    hist["qty"]       = pd.to_numeric(hist.get("qty", 1), errors="coerce").fillna(1.0)
    hist["timestamp"] = pd.to_datetime(hist.get("timestamp", pd.NaT), errors="coerce")
    hist = hist.dropna(subset=["timestamp"])

    hist["week"] = hist["timestamp"].dt.to_period("W")
    weekly_hist  = (
        hist.groupby(["item_name", "week"])["qty"].sum().reset_index()
    )
    hist_avg = weekly_hist.groupby("item_name")["qty"].mean()

    # First-week forecast per item
    first_week = (
        forecast_df.sort_values("week_start")
        .groupby("item_name")
        .first()["predicted_qty"]
    )

    for item_name in first_week.index:
        if item_name not in hist_avg.index:
            continue

        h_val = float(hist_avg[item_name])
        p_val = float(first_week[item_name])

        if h_val <= 0:
            continue

        change = (p_val - h_val) / h_val

        # Spike
        if change >= spike_thresh:
            alerts.append({
                "item_name":  item_name,
                "alert_type": "spike",
                "severity":   "warning",
                "message_th": (
                    f"{item_name}: คาดว่าจะขายเพิ่มขึ้น {change*100:.0f}% "
                    f"({h_val:.0f} → {p_val:.0f} ต่อสัปดาห์) — ควรเตรียมวัตถุดิบเพิ่ม"
                ),
            })

        # Low stock risk (demand >> historical supply proxy)
        if p_val > 2 * h_val:
            alerts.append({
                "item_name":  item_name,
                "alert_type": "low_stock_risk",
                "severity":   "critical",
                "message_th": (
                    f"{item_name}: ความต้องการคาดว่าจะสูงกว่าปกติมาก "
                    f"({p_val:.0f} vs ปกติ {h_val:.0f}/สัปดาห์) — ตรวจสอบสต็อกวัตถุดิบด่วน"
                ),
            })

        # Slow mover — consecutive weekly decline
        item_hist = (
            weekly_hist[weekly_hist["item_name"] == item_name]
            .sort_values("week")
            .tail(slow_weeks)
        )
        if len(item_hist) >= slow_weeks:
            vals = item_hist["qty"].values
            first_val = float(vals[0])
            last_val  = float(vals[-1])
            if first_val > 0:
                trend = (last_val - first_val) / first_val
                if trend <= -slow_thresh:
                    alerts.append({
                        "item_name":  item_name,
                        "alert_type": "slow_mover",
                        "severity":   "info",
                        "message_th": (
                            f"{item_name}: ยอดขายลดลง {abs(trend)*100:.0f}% "
                            f"ใน {slow_weeks} สัปดาห์ — พิจารณาปรับราคาหรือโปรโมชัน"
                        ),
                    })

    _sev_order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(alerts, key=lambda x: _sev_order.get(x["severity"], 3))

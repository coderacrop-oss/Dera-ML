import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from crm_api.app import create_app
from crm_api.settings import Settings

KEY = "test-key"
CAL_TOKEN = "cal-token"
HDR = {"X-API-Key": KEY}

ACTIONS = {1: "Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ", 2: "Standard Retention", 3: "VIP Early Access"}
WINDOWS = {1: "Off-Peak Window", 2: "Peak Window", 3: "Peak Window"}


def make_schedule_rows(n=12):
    rows = []
    for i in range(n):
        wave = 1 + i % 3
        rows.append(dict(
            customer_id=f"C{i:03d}", segment_name="1. Champions", wave=wave,
            recommended_action=ACTIONS[wave],
            send_datetime=f"2026-12-0{1 + i % 4} 0{7 + (i % 3)}:00:00",
            send_window=WINDOWS[wave], send_time_hint="Before Peak Window", channel="Email/SMS",
            message="ข้อเสนอพิเศษ, ลด 10%; ดูเลย" if i == 0 else f"msg {i}",
            offer="ลด 10%", expected_profit=10.0 + i, should_send=1))
    return rows


def write_fixture_month(dir_, month="2026-12", *, schedule_rows=None, segments=True):
    schedule = pd.DataFrame(schedule_rows if schedule_rows is not None else make_schedule_rows())
    schedule.to_csv(dir_ / f"campaign_schedule_{month}.csv", index=False)
    pd.DataFrame([dict(customer_id="H001", segment_name="1. Champions", recommended_action=ACTIONS[1],
                       offer="x", expected_profit=5.0, holdout_reason="5% control"),
                  dict(customer_id="H002", segment_name="1. Champions", recommended_action=ACTIONS[2],
                       offer="x", expected_profit=6.0, holdout_reason="5% control")]
                 ).to_csv(dir_ / f"ab_holdout_{month}.csv", index=False)
    wave_counts = schedule["wave"].value_counts().to_dict() if len(schedule) else {}
    total = round(float(schedule["expected_profit"].sum()), 2) if len(schedule) else 0.0
    summary = {
        "month": month, "total_scheduled": len(schedule), "total_holdout": 2, "holdout_rate_pct": 5.0,
        "peak_hours_detected": [11, 12], "off_peak_hours_detected": [8, 9],
        "wave_breakdown": {"wave_1_off_peak_traffic": int(wave_counts.get(1, 0)),
                           "wave_2_peak_upsell": int(wave_counts.get(2, 0)),
                           "wave_3_vip_early": int(wave_counts.get(3, 0))},
        "send_window_breakdown": {}, "daily_send_counts": {},
        "total_expected_profit": total, "avg_profit_per_customer": round(total / max(1, len(schedule)), 2),
    }
    (dir_ / f"campaign_summary_{month}.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    if segments:
        (dir_ / f"segments_{month}.json").write_text(json.dumps({
            "generated_at": "2026-11-03T03:00:00+00:00", "gate_status": "PASSED",
            "segments": [{"id": 1, "name": "1. Champions", "customers": 516, "broadcast": 12, "suppressed": 504,
                          "recommended_promo": "VIP Early Access", "action_breakdown": {"VIP Early Access": 12},
                          "unit_economics": {"net_profit_per_head": 206.27, "total_projected_profit": 106433.28,
                                             "l4_total_projected_profit": 5000.5, "financial_status": "PROFITABLE"}}],
            "wave_weeks": {"1": [1, 3], "2": [2, 4], "3": [1]}}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path):
    write_fixture_month(tmp_path)
    return tmp_path


@pytest.fixture
def client(data_dir):
    return TestClient(create_app(Settings(data_dir=data_dir, api_key=KEY, calendar_token=CAL_TOKEN)))

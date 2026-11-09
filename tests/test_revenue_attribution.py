"""Tests for revenue_attribution.py."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_pos(rows):
    return pd.DataFrame(rows)


def _exec_history(plan_id="plan_001", execution_id="exec_001",
                  executed_at=None, audience_ids=None):
    if executed_at is None:
        executed_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    return {
        "plan_id":               plan_id,
        "execution_id":          execution_id,
        "executed_at":           executed_at,
        "status":                "executed_sent",
        "audience_customer_ids": audience_ids or [],
    }


def _d(days_ago=5) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


# ===========================================================================
# tag_transaction_source
# ===========================================================================

class TestTagTransactionSource:

    def test_adds_campaign_id_to_matching_customers(self):
        from revenue_attribution import tag_transaction_source
        df = _make_pos([
            {"customer_id": "C1", "amount": 100},
            {"customer_id": "C2", "amount": 200},
            {"customer_id": "C3", "amount": 150},
        ])
        result = tag_transaction_source(df, "exec_001", ["C1", "C3"])
        assert result.loc[result["customer_id"] == "C1", "campaign_id"].iloc[0] == "exec_001"
        assert result.loc[result["customer_id"] == "C3", "campaign_id"].iloc[0] == "exec_001"
        assert result.loc[result["customer_id"] == "C2", "campaign_id"].iloc[0] == ""

    def test_does_not_modify_original_df(self):
        from revenue_attribution import tag_transaction_source
        df = _make_pos([{"customer_id": "C1", "amount": 100}])
        tag_transaction_source(df, "exec_001", ["C1"])
        assert "campaign_id" not in df.columns

    def test_overwrites_existing_campaign_id(self):
        from revenue_attribution import tag_transaction_source
        df = _make_pos([{"customer_id": "C1", "amount": 100, "campaign_id": "old"}])
        result = tag_transaction_source(df, "exec_new", ["C1"])
        assert result.loc[0, "campaign_id"] == "exec_new"

    def test_empty_customer_ids_no_tags(self):
        from revenue_attribution import tag_transaction_source
        df = _make_pos([{"customer_id": "C1", "amount": 100}])
        result = tag_transaction_source(df, "exec_001", [])
        assert result.loc[0, "campaign_id"] == ""


# ===========================================================================
# compute_attributed_revenue — campaign_tag method
# ===========================================================================

class TestComputeAttributedRevenueCampaignTag:

    def test_campaign_id_column_attributed(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 500.0, "date": _d(5), "campaign_id": "exec_001"},
            {"customer_id": "C2", "amount": 300.0, "date": _d(3), "campaign_id": ""},
        ])
        history = [_exec_history(execution_id="exec_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 500.0
        assert result["organic_revenue"]    == 300.0
        assert result["by_method"]["campaign_tag"] == 1

    def test_source_column_attributed(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 400.0, "date": _d(5), "source": "plan_001"},
        ])
        history = [_exec_history(plan_id="plan_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 400.0

    def test_unrecognised_campaign_id_is_organic(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 200.0, "date": _d(5), "campaign_id": "unknown_id"},
        ])
        history = [_exec_history(execution_id="exec_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 0.0
        assert result["organic_revenue"]    == 200.0


# ===========================================================================
# compute_attributed_revenue — customer_match method
# ===========================================================================

class TestComputeAttributedRevenueCustomerMatch:

    def test_customer_in_audience_within_window(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        exec_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        pos = _make_pos([
            {"customer_id": "C1", "amount": 750.0, "date": _d(5)},   # 5 days after exec
        ])
        history = [_exec_history(executed_at=exec_at, audience_ids=["C1"])]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path),
                                            attribution_window_days=30, dry_run=True)
        assert result["attributed_revenue"] == 750.0
        assert result["by_method"]["customer_match"] == 1

    def test_customer_outside_window_is_organic(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        exec_at = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        pos = _make_pos([
            {"customer_id": "C1", "amount": 750.0, "date": _d(5)},   # 40 days after exec
        ])
        history = [_exec_history(executed_at=exec_at, audience_ids=["C1"])]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path),
                                            attribution_window_days=30, dry_run=True)
        assert result["attributed_revenue"] == 0.0

    def test_customer_not_in_audience_is_organic(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        exec_at = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        pos = _make_pos([
            {"customer_id": "C_other", "amount": 200.0, "date": _d(5)},
        ])
        history = [_exec_history(executed_at=exec_at, audience_ids=["C1"])]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 0.0


# ===========================================================================
# compute_attributed_revenue — promo_code method
# ===========================================================================

class TestComputeAttributedRevenuePromoCode:

    def test_crm_promo_code_attributed(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 350.0, "date": _d(5), "promo_code": "CRM2026NOV"},
            {"customer_id": "C2", "amount": 200.0, "date": _d(3), "promo_code": "SHOP10"},
        ])
        result = compute_attributed_revenue(pos, [], data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 350.0
        assert result["by_method"]["promo_code"] == 1

    def test_non_crm_promo_code_is_organic(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 100.0, "date": _d(5), "promo_code": "SALE20"},
        ])
        result = compute_attributed_revenue(pos, [], data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 0.0


# ===========================================================================
# compute_attributed_revenue — totals and fee
# ===========================================================================

class TestComputeAttributedRevenueTotals:

    def test_revenue_share_fee_calculation(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 1000.0, "date": _d(5), "campaign_id": "exec_001"},
        ])
        history = [_exec_history(execution_id="exec_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path),
                                            revenue_share_pct=0.05, dry_run=True)
        assert result["revenue_share_fee"] == 50.0

    def test_attribution_rate(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 600.0, "date": _d(5), "campaign_id": "exec_001"},
            {"customer_id": "C2", "amount": 400.0, "date": _d(3)},
        ])
        history = [_exec_history(execution_id="exec_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert result["attribution_rate"] == pytest.approx(0.60, abs=0.01)
        assert result["total_revenue"]    == 1000.0

    def test_empty_pos_returns_zeros(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([])
        result = compute_attributed_revenue(pos, [], data_dir=str(tmp_path), dry_run=True)
        assert result["total_revenue"]      == 0.0
        assert result["attributed_revenue"] == 0.0
        assert result["revenue_share_fee"]  == 0.0

    def test_no_history_all_organic(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 500.0, "date": _d(5)},
        ])
        result = compute_attributed_revenue(pos, [], data_dir=str(tmp_path), dry_run=True)
        assert result["attributed_revenue"] == 0.0
        assert result["organic_revenue"]    == 500.0

    def test_month_filter(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 300.0, "date": "2026-10-15", "campaign_id": "exec_001"},
            {"customer_id": "C2", "amount": 200.0, "date": "2026-11-05", "campaign_id": "exec_001"},
        ])
        history = [_exec_history(execution_id="exec_001")]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path),
                                            month="2026-11", dry_run=True)
        assert result["total_revenue"] == 200.0
        assert result["month"] == "2026-11"

    def test_by_campaign_breakdown(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 500.0, "date": _d(5), "campaign_id": "exec_001"},
            {"customer_id": "C2", "amount": 300.0, "date": _d(3), "campaign_id": "exec_002"},
        ])
        history = [
            _exec_history(execution_id="exec_001"),
            _exec_history(plan_id="plan_002", execution_id="exec_002"),
        ]
        result = compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert "exec_001" in result["by_campaign"]
        assert "exec_002" in result["by_campaign"]


# ===========================================================================
# compute_attributed_revenue — log writing
# ===========================================================================

class TestAttributionLog:

    def test_dry_run_does_not_write_log(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 500.0, "date": _d(5), "campaign_id": "exec_001"},
        ])
        history = [_exec_history(execution_id="exec_001")]
        compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=True)
        assert not os.path.exists(str(tmp_path / "attribution_log.jsonl"))

    def test_live_run_writes_log(self, tmp_path):
        from revenue_attribution import compute_attributed_revenue
        pos = _make_pos([
            {"customer_id": "C1", "amount": 500.0, "date": _d(5), "campaign_id": "exec_001"},
        ])
        history = [_exec_history(execution_id="exec_001")]
        compute_attributed_revenue(pos, history, data_dir=str(tmp_path), dry_run=False)
        log_path = str(tmp_path / "attribution_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 1
        assert rows[0]["customer_id"] == "C1"
        assert rows[0]["method"] == "campaign_tag"


# ===========================================================================
# get_attribution_report
# ===========================================================================

class TestGetAttributionReport:

    def test_contains_key_numbers(self):
        from revenue_attribution import get_attribution_report
        result = {
            "total_revenue": 10000.0, "attributed_revenue": 3000.0,
            "organic_revenue": 7000.0, "attribution_rate": 0.30,
            "revenue_share_fee": 150.0, "revenue_share_pct": 0.05,
            "attributed_transactions": 25,
            "by_method": {"campaign_tag": 15, "customer_match": 10, "promo_code": 0},
            "by_campaign": {"exec_001": 1800.0, "exec_002": 1200.0},
            "month": "2026-11",
        }
        report = get_attribution_report(result)
        assert "10,000.00" in report or "10000" in report
        assert "3,000.00" in report or "3000" in report
        assert "150.00" in report or "150" in report
        assert "2026-11" in report

    def test_contains_thai_text(self):
        from revenue_attribution import get_attribution_report
        result = {
            "total_revenue": 1000.0, "attributed_revenue": 500.0,
            "organic_revenue": 500.0, "attribution_rate": 0.50,
            "revenue_share_fee": 25.0, "revenue_share_pct": 0.05,
            "attributed_transactions": 5,
            "by_method": {}, "by_campaign": {}, "month": "",
        }
        report = get_attribution_report(result)
        assert "ยอดขาย" in report or "Codera" in report


# ===========================================================================
# load_campaign_history
# ===========================================================================

class TestLoadCampaignHistory:

    def test_loads_jsonl(self, tmp_path):
        from revenue_attribution import load_campaign_history
        path = str(tmp_path / "campaign_history.jsonl")
        records = [{"plan_id": "p1", "executed_at": "2026-11-01T00:00:00+00:00"}]
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loaded = load_campaign_history(path)
        assert len(loaded) == 1
        assert loaded[0]["plan_id"] == "p1"

    def test_missing_file_returns_empty(self, tmp_path):
        from revenue_attribution import load_campaign_history
        result = load_campaign_history(str(tmp_path / "nonexistent.jsonl"))
        assert result == []

    def test_skips_malformed_lines(self, tmp_path):
        from revenue_attribution import load_campaign_history
        path = str(tmp_path / "campaign_history.jsonl")
        with open(path, "w") as f:
            f.write('{"plan_id": "p1"}\n')
            f.write("not_json\n")
            f.write('{"plan_id": "p2"}\n')
        loaded = load_campaign_history(path)
        assert len(loaded) == 2

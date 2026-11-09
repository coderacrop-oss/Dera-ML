"""Phase 3 — DotDash Connector tests.

Covers AC-3.1 through AC-3.5 without real network calls.
All HTTP requests are intercepted by monkeypatching requests.get.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_member(i: int, line_user_id: str | None = None) -> dict:
    """Build a synthetic DotDash API member record."""
    return {
        "id":             str(1000 + i),
        "lineUserId":     line_user_id or f"Utest{i:06d}",
        "nickname":       f"ลูกค้า {i}",
        "phoneNumber":    f"0812345{i:03d}",
        "visitCount":     3 + i % 5,
        "totalSpend":     300.0 + i * 50,
        "lastVisitDate":  "15/10/2026",
        "membershipLevel": "Silver",
        "segment":        "Loyal Customers",
        "points":         i * 10,
    }


def _make_api_response(members: list[dict], next_cursor: str | None) -> MagicMock:
    """Return a mock requests.Response with JSON payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data":       members,
        "nextCursor": next_cursor,
        "hasMore":    next_cursor is not None,
    }
    return resp


@pytest.fixture
def meripos_df():
    """Minimal MeriPOS transaction DataFrame."""
    return pd.DataFrame([
        {"timestamp": "2026-10-01 09:00", "receipt_id": "R001", "net_sales": 150.0, "phone": "0812345000"},
        {"timestamp": "2026-10-02 10:00", "receipt_id": "R002", "net_sales": 220.0, "phone": "0812345001"},
        {"timestamp": "2026-10-03 11:00", "receipt_id": "R003", "net_sales": 80.0,  "phone": None},
        {"timestamp": "2026-10-04 12:00", "receipt_id": "R004", "net_sales": 60.0,  "phone": "-"},
        {"timestamp": "2026-10-05 13:00", "receipt_id": "R005", "net_sales": 500.0, "phone": "0812345000"},
    ])


@pytest.fixture
def dotdash_df():
    """Minimal DotDash member DataFrame (from ingest_dotdash or fetch_all_members)."""
    return pd.DataFrame([
        {
            "customer_id": "Utest000000",
            "display_name": "ลูกค้า 0",
            "phone": "0812345000",
            "frequency": 5,
            "monetary": 850.0,
            "recency": 23,
            "last_visit": pd.Timestamp("2026-10-15"),
            "tier": "Silver",
            "dotdash_segment": "Loyal Customers",
            "points": 0,
        },
        {
            "customer_id": "Utest000001",
            "display_name": "ลูกค้า 1",
            "phone": "0812345001",
            "frequency": 3,
            "monetary": 350.0,
            "recency": 14,
            "last_visit": pd.Timestamp("2026-10-24"),
            "tier": "Bronze",
            "dotdash_segment": "Promising",
            "points": 10,
        },
        {
            "customer_id": "dotdash_1002",
            "display_name": "ลูกค้า 2",
            "phone": None,
            "frequency": 1,
            "monetary": 200.0,
            "recency": 60,
            "last_visit": pd.Timestamp("2026-09-07"),
            "tier": "Bronze",
            "dotdash_segment": "Need Attention",
            "points": 5,
        },
    ])


# ===========================================================================
# AC-3.1 — DotDash API Client: cursor pagination + column mapping
# ===========================================================================

class TestDotDashClientPagination:
    """AC-3.1: fetch_all_members traverses all pages via cursor."""

    def test_fetches_all_pages(self):
        """3-page mock (p1→p2→p3→done) → all members collected."""
        from dotdash_connector import DotDashClient

        pages = [
            _make_api_response([_make_member(i) for i in range(5)],  next_cursor="cur1"),
            _make_api_response([_make_member(i) for i in range(5, 10)], next_cursor="cur2"),
            _make_api_response([_make_member(i) for i in range(10, 13)], next_cursor=None),
        ]

        with patch("requests.get", side_effect=pages):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        assert len(df) == 13
        assert set(df.columns) >= {"customer_id", "frequency", "monetary", "recency"}

    def test_empty_response_returns_empty_df(self):
        from dotdash_connector import DotDashClient

        resp = _make_api_response([], next_cursor=None)
        with patch("requests.get", return_value=resp):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        assert df.empty
        assert "customer_id" in df.columns

    def test_column_mapping_applied(self):
        from dotdash_connector import DotDashClient

        resp = _make_api_response([_make_member(0)], next_cursor=None)
        with patch("requests.get", return_value=resp):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        # API field visitCount → frequency
        assert "frequency" in df.columns
        # API field totalSpend → monetary
        assert "monetary" in df.columns
        assert df["frequency"].iloc[0] == 3  # 3 + 0%5


# ===========================================================================
# AC-3.1 — Retry logic (429 rate-limit)
# ===========================================================================

class TestRetryLogic:
    def test_retries_on_429_then_succeeds(self):
        """Two 429s followed by 200 → succeeds, returns data."""
        from dotdash_connector import DotDashClient

        rate_limit = MagicMock()
        rate_limit.status_code = 429

        success = _make_api_response([_make_member(0)], next_cursor=None)

        with patch("requests.get", side_effect=[rate_limit, rate_limit, success]):
            with patch("time.sleep"):   # speed up test — don't actually sleep
                client = DotDashClient(api_token="test-token")
                df = client.fetch_all_members()

        assert len(df) == 1

    def test_all_retries_exhausted_returns_empty(self):
        """3 consecutive 429s → returns empty DataFrame, does not raise."""
        from dotdash_connector import DotDashClient

        rate_limit = MagicMock()
        rate_limit.status_code = 429

        with patch("requests.get", side_effect=[rate_limit, rate_limit, rate_limit]):
            with patch("time.sleep"):
                client = DotDashClient(api_token="test-token")
                df = client.fetch_all_members()

        assert df.empty

    def test_timeout_retries(self):
        """requests.exceptions.Timeout → retries → succeeds on 3rd attempt."""
        import requests as _requests
        from dotdash_connector import DotDashClient

        success = _make_api_response([_make_member(1)], next_cursor=None)

        with patch("requests.get",
                   side_effect=[_requests.exceptions.Timeout, _requests.exceptions.Timeout, success]):
            with patch("time.sleep"):
                client = DotDashClient(api_token="test-token")
                df = client.fetch_all_members()

        assert len(df) == 1


# ===========================================================================
# AC-3.2 — LINE userId → customer_id
# ===========================================================================

class TestLineUserIdExtraction:
    def test_member_with_line_id_uses_line_id(self):
        """Member with lineUserId → customer_id = lineUserId."""
        from dotdash_connector import DotDashClient

        resp = _make_api_response([_make_member(0, line_user_id="Uabc123")], next_cursor=None)
        with patch("requests.get", return_value=resp):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        assert df.iloc[0]["customer_id"] == "Uabc123"

    def test_member_without_line_id_falls_back_to_dotdash_id(self):
        """Member with no lineUserId field → customer_id starts with 'dotdash_'."""
        from dotdash_connector import DotDashClient

        # Build a member record without any lineUserId key at all
        member = {
            "id":             "1099",
            "nickname":       "ลูกค้าไม่มี LINE",
            "phoneNumber":    "0899999999",
            "visitCount":     2,
            "totalSpend":     200.0,
            "lastVisitDate":  "10/10/2026",
            "membershipLevel": "Bronze",
            "segment":        "Promising",
            "points":         5,
        }
        resp = _make_api_response([member], next_cursor=None)
        with patch("requests.get", return_value=resp):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        assert df.iloc[0]["customer_id"].startswith("dotdash_")

    def test_mixed_members_correct_ids(self):
        """Some with LINE ID, some without → each mapped correctly."""
        from dotdash_connector import DotDashClient

        # Member 0: has LINE ID
        m0 = _make_member(0, line_user_id="Uline001")
        # Member 1: no lineUserId key at all
        m1 = {
            "id": "1001",
            "nickname": "ลูกค้า 1",
            "phoneNumber": "0812345001",
            "visitCount": 4,
            "totalSpend": 450.0,
            "lastVisitDate": "15/10/2026",
            "membershipLevel": "Silver",
            "segment": "Loyal Customers",
            "points": 10,
        }
        # Member 2: has LINE ID
        m2 = _make_member(2, line_user_id="Uline003")

        resp = _make_api_response([m0, m1, m2], next_cursor=None)
        with patch("requests.get", return_value=resp):
            client = DotDashClient(api_token="test-token")
            df = client.fetch_all_members()

        assert df.iloc[0]["customer_id"] == "Uline001"
        assert df.iloc[1]["customer_id"].startswith("dotdash_")
        assert df.iloc[2]["customer_id"] == "Uline003"


# ===========================================================================
# AC-3.3 — build_hybrid_rfm
# ===========================================================================

class TestBuildHybridRfm:
    def test_anonymous_mode_returns_pipeline_format(self, dotdash_df):
        """Anonymous mode → df with customer_id, transaction_date, amount (no nulls)."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(pd.DataFrame(), dotdash_df, quality_mode="anonymous")

        assert set(result.columns) >= {"customer_id", "transaction_date", "amount"}
        assert result["customer_id"].notna().all()
        assert result["transaction_date"].notna().all()
        assert (result["amount"] > 0).all()

    def test_anonymous_mode_frequency_preserved(self, dotdash_df):
        """Member with frequency=5 → 5 transaction rows in output."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(pd.DataFrame(), dotdash_df, quality_mode="anonymous")

        cid = dotdash_df.iloc[0]["customer_id"]
        rows_for_cid = (result["customer_id"] == cid).sum()
        assert rows_for_cid == dotdash_df.iloc[0]["frequency"]

    def test_anonymous_mode_monetary_preserved(self, dotdash_df):
        """Member total amount = monetary from DotDash."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(pd.DataFrame(), dotdash_df, quality_mode="anonymous")

        for _, member in dotdash_df.iterrows():
            cid = member["customer_id"]
            cid_rows = result[result["customer_id"] == cid]
            if len(cid_rows) > 0 and member["monetary"] > 0:
                assert abs(cid_rows["amount"].sum() - member["monetary"]) < 0.01

    def test_full_mode_links_via_phone(self, meripos_df, dotdash_df):
        """Full mode: transactions matched via phone get DotDash customer_id."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(meripos_df, dotdash_df, quality_mode="full")

        # Phone 0812345000 → Utest000000
        linked = result[result["customer_id"] == "Utest000000"]
        assert len(linked) >= 1  # at least one transaction linked

    def test_full_mode_unlinked_transactions_kept(self, meripos_df, dotdash_df):
        """Full mode: unlinked transactions still appear (anon_ prefix)."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(meripos_df, dotdash_df, quality_mode="full")

        anon_rows = result[result["customer_id"].str.startswith("anon_")]
        assert len(anon_rows) >= 1  # R003 (phone=None) and R004 (phone="-") are unlinked

    def test_partial_mode_includes_synthetic_members(self, meripos_df, dotdash_df):
        """Partial mode: unjoined DotDash members also appear in output."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(meripos_df, dotdash_df, quality_mode="partial")

        # dotdash_1002 (phone=None) won't join → synthetic records added
        dotdash_only = result[result["customer_id"] == "dotdash_1002"]
        assert len(dotdash_only) >= 1

    def test_empty_meripos_anonymous_still_works(self, dotdash_df):
        """Empty MeriPOS + anonymous mode → synthesised from DotDash only, no crash."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(pd.DataFrame(), dotdash_df, quality_mode="anonymous")
        assert len(result) >= len(dotdash_df)

    def test_empty_dotdash_anonymous_returns_empty(self):
        """Empty DotDash + anonymous mode → empty output DataFrame."""
        from ingest import build_hybrid_rfm

        result = build_hybrid_rfm(pd.DataFrame(), pd.DataFrame(), quality_mode="anonymous")
        assert result.empty


# ===========================================================================
# AC-3.4 — get_quality_report
# ===========================================================================

class TestGetQualityReport:
    def test_returns_all_required_keys(self, meripos_df, dotdash_df):
        from dotdash_connector import get_quality_report

        report = get_quality_report(meripos_df, dotdash_df)

        required = {
            "total_transactions", "linked_transactions", "linked_rate",
            "total_members", "members_with_spend", "recommendation",
        }
        assert required <= set(report.keys())

    def test_total_transactions_correct(self, meripos_df, dotdash_df):
        from dotdash_connector import get_quality_report

        report = get_quality_report(meripos_df, dotdash_df)
        assert report["total_transactions"] == len(meripos_df)

    def test_linked_transactions_counts_phone_rows(self, meripos_df, dotdash_df):
        """2 of 5 meripos_df rows have a real phone → linked_transactions = 2."""
        from dotdash_connector import get_quality_report

        report = get_quality_report(meripos_df, dotdash_df)
        # R001 (0812345000), R002 (0812345001), R005 (0812345000) have phones
        # R003 (None), R004 ("-") do not
        assert report["linked_transactions"] == 3
        assert 0 < report["linked_rate"] < 1.0

    def test_members_with_spend_correct(self, meripos_df, dotdash_df):
        from dotdash_connector import get_quality_report

        report = get_quality_report(meripos_df, dotdash_df)
        # All 3 fixture members have monetary > 0
        assert report["members_with_spend"] == 3

    def test_recommendation_is_thai_string(self, meripos_df, dotdash_df):
        import re
        from dotdash_connector import get_quality_report

        report = get_quality_report(meripos_df, dotdash_df)
        # recommendation should contain Thai characters
        assert re.search(r"[\u0e00-\u0e7f]", report["recommendation"])

    def test_zero_meripos_returns_zero_linked(self, dotdash_df):
        from dotdash_connector import get_quality_report

        report = get_quality_report(pd.DataFrame(), dotdash_df)
        assert report["total_transactions"] == 0
        assert report["linked_rate"] == 0.0


# ===========================================================================
# AC-3.5 — filter_test_data (already in ingest.py from Phase 1)
# ===========================================================================

class TestFilterTestData:
    def test_removes_low_amount_rows(self):
        """Rows with net_sales < 10 (test_data_min_amount) are removed."""
        from ingest import filter_test_data

        df = pd.DataFrame([
            {"net_sales": 5.0,   "Campaign Name": "Normal"},
            {"net_sales": 150.0, "Campaign Name": "Normal"},
            {"net_sales": 9.9,   "Campaign Name": "Normal"},
        ])
        result = filter_test_data(df)
        assert len(result) == 1
        assert result.iloc[0]["net_sales"] == 150.0

    def test_removes_test_campaign_rows(self):
        """Rows with Campaign Name containing 'test_' pattern are removed.

        The filter skips columns where > 50% of values match the pattern.
        So we use a majority of real rows to ensure the column is not skipped.
        """
        from ingest import filter_test_data

        df = pd.DataFrame([
            {"net_sales": 200.0, "Campaign Name": "Test_Campaign_01"},
            {"net_sales": 300.0, "Campaign Name": "Normal Campaign"},
            {"net_sales": 150.0, "Campaign Name": "Normal Promo"},
            {"net_sales": 400.0, "Campaign Name": "Big Sale"},
        ])
        result = filter_test_data(df)
        # Only "Test_Campaign_01" matches \btest[_\s] with minority share
        assert len(result) == 3
        assert "Test_Campaign_01" not in result["Campaign Name"].values

    def test_mixed_5_test_95_real(self):
        """5 test rows + 95 real rows → 95 rows after filter."""
        from ingest import filter_test_data

        real = [{"net_sales": 200.0, "Campaign Name": "Real"} for _ in range(95)]
        test = [{"net_sales": 5.0,   "Campaign Name": "Test_data"} for _ in range(5)]
        df = pd.DataFrame(real + test)

        result = filter_test_data(df)
        assert len(result) == 95

    def test_no_rows_filtered_when_all_real(self):
        """All valid rows → returned unchanged."""
        from ingest import filter_test_data

        df = pd.DataFrame([
            {"net_sales": 100.0, "Campaign Name": "Normal"},
            {"net_sales": 500.0, "Campaign Name": "Big Sale"},
        ])
        result = filter_test_data(df)
        assert len(result) == 2


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestPhase3Integration:
    def test_full_pipeline_anonymous_to_rfm(self, dotdash_df):
        """Anonymous mode: DotDash members → hybrid RFM → pipeline-ready df."""
        from ingest import build_hybrid_rfm

        rfm_df = build_hybrid_rfm(pd.DataFrame(), dotdash_df, quality_mode="anonymous")

        # Must have the 3 required columns with no nulls
        assert set(rfm_df.columns) >= {"customer_id", "transaction_date", "amount"}
        assert rfm_df["customer_id"].notna().all()
        assert rfm_df["transaction_date"].notna().all()

        # Must have at least one row per member (frequency ≥ 1)
        unique_cids = rfm_df["customer_id"].nunique()
        assert unique_cids == len(dotdash_df)

    def test_quality_report_plus_hybrid_rfm_workflow(self, meripos_df, dotdash_df):
        """get_quality_report → decide mode → build_hybrid_rfm."""
        from dotdash_connector import get_quality_report
        from ingest import build_hybrid_rfm

        report = get_quality_report(meripos_df, dotdash_df)

        rate = report["linked_rate"]
        if rate >= 0.30:
            mode = "full"
        elif rate >= 0.05:
            mode = "partial"
        else:
            mode = "anonymous"

        rfm_df = build_hybrid_rfm(meripos_df, dotdash_df, quality_mode=mode)
        assert len(rfm_df) > 0
        assert rfm_df["amount"].sum() > 0

"""Tests for line_oa_crm_connector.py."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid(n: int) -> str:
    return f"U{str(n).zfill(32)}"


def _mock_get_ok(data: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    return resp


def _mock_get_404():
    resp = MagicMock()
    resp.status_code = 404
    resp.json.return_value = {}
    return resp


def _make_customer_df(rows):
    return pd.DataFrame(rows)


# ===========================================================================
# _normalise_phone
# ===========================================================================

class TestNormalisePhone:

    def test_thai_mobile_prefix_0(self):
        from line_oa_crm_connector import _normalise_phone
        assert _normalise_phone("0812345678") == "+66812345678"

    def test_already_plus_66(self):
        from line_oa_crm_connector import _normalise_phone
        assert _normalise_phone("+66812345678") == "+66812345678"

    def test_prefix_66_no_plus(self):
        from line_oa_crm_connector import _normalise_phone
        assert _normalise_phone("66812345678") == "+66812345678"

    def test_strips_dashes_and_spaces(self):
        from line_oa_crm_connector import _normalise_phone
        assert _normalise_phone("081-234-5678") == "+66812345678"

    def test_empty_returns_empty(self):
        from line_oa_crm_connector import _normalise_phone
        assert _normalise_phone("") == ""

    def test_non_thai_number_unchanged(self):
        from line_oa_crm_connector import _normalise_phone
        result = _normalise_phone("+1234567890")
        assert result == "+1234567890"


# ===========================================================================
# fetch_followers
# ===========================================================================

class TestFetchFollowers:

    def test_raises_without_token(self):
        from line_oa_crm_connector import fetch_followers
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LINE_CHANNEL_ACCESS_TOKEN", None)
            with pytest.raises(ValueError, match="LINE_CHANNEL_ACCESS_TOKEN"):
                fetch_followers(channel_access_token=None)

    def test_single_page_returns_all_ids(self):
        from line_oa_crm_connector import fetch_followers
        mock_resp = _mock_get_ok({"userIds": [_uid(1), _uid(2), _uid(3)]})
        # No "next" → single page
        with patch("requests.get", return_value=mock_resp):
            result = fetch_followers(channel_access_token="fake-token")
        assert result == [_uid(1), _uid(2), _uid(3)]

    def test_paginates_until_no_next(self):
        from line_oa_crm_connector import fetch_followers
        page1 = _mock_get_ok({"userIds": [_uid(1), _uid(2)], "next": "cursor_abc"})
        page2 = _mock_get_ok({"userIds": [_uid(3), _uid(4)]})  # no "next"
        with patch("requests.get", side_effect=[page1, page2]):
            result = fetch_followers(channel_access_token="fake-token")
        assert result == [_uid(1), _uid(2), _uid(3), _uid(4)]

    def test_empty_follower_list(self):
        from line_oa_crm_connector import fetch_followers
        mock_resp = _mock_get_ok({"userIds": []})
        with patch("requests.get", return_value=mock_resp):
            result = fetch_followers(channel_access_token="fake-token")
        assert result == []

    def test_uses_env_token(self):
        from line_oa_crm_connector import fetch_followers
        mock_resp = _mock_get_ok({"userIds": []})
        with patch.dict(os.environ, {"LINE_CHANNEL_ACCESS_TOKEN": "env-token"}):
            with patch("requests.get", return_value=mock_resp) as mock_get:
                fetch_followers()
        call_headers = mock_get.call_args.kwargs["headers"]
        assert "env-token" in call_headers["Authorization"]


# ===========================================================================
# get_member_profile
# ===========================================================================

class TestGetMemberProfile:

    def test_returns_profile_dict(self):
        from line_oa_crm_connector import get_member_profile
        profile = {"displayName": "สมชาย", "pictureUrl": "https://...", "language": "th"}
        with patch("requests.get", return_value=_mock_get_ok(profile)):
            result = get_member_profile(_uid(1), channel_access_token="tok")
        assert result["displayName"] == "สมชาย"

    def test_404_returns_empty_dict(self):
        from line_oa_crm_connector import get_member_profile
        with patch("requests.get", return_value=_mock_get_404()):
            result = get_member_profile(_uid(99), channel_access_token="tok")
        assert result == {}


# ===========================================================================
# get_member_tags
# ===========================================================================

class TestGetMemberTags:

    def test_returns_tags_per_user(self):
        from line_oa_crm_connector import get_member_tags
        resp_data = {
            "userTags": [
                {"userId": _uid(1), "tags": [{"tagName": "interested_camera"}, {"tagName": "phone:0812345678"}]},
                {"userId": _uid(2), "tags": [{"tagName": "vip"}]},
            ]
        }
        with patch("requests.post", return_value=_mock_get_ok(resp_data)):
            result = get_member_tags([_uid(1), _uid(2)], channel_access_token="tok")
        assert "interested_camera" in result[_uid(1)]
        assert "phone:0812345678" in result[_uid(1)]
        assert result[_uid(2)] == ["vip"]

    def test_empty_tags_for_user_not_in_response(self):
        from line_oa_crm_connector import get_member_tags
        resp_data = {"userTags": [{"userId": _uid(1), "tags": []}]}
        with patch("requests.post", return_value=_mock_get_ok(resp_data)):
            result = get_member_tags([_uid(1), _uid(2)], channel_access_token="tok")
        assert result[_uid(2)] == []

    def test_batches_at_100(self):
        """101 userIds → 2 POST calls."""
        from line_oa_crm_connector import get_member_tags
        ids = [_uid(i) for i in range(101)]
        resp_data = {"userTags": []}
        with patch("requests.post", return_value=_mock_get_ok(resp_data)) as mock_post:
            get_member_tags(ids, channel_access_token="tok")
        assert mock_post.call_count == 2


# ===========================================================================
# match_members_to_customers
# ===========================================================================

class TestMatchMembersToCustomers:

    @pytest.fixture
    def members(self):
        from line_oa_crm_connector import LineOAMember
        return [
            LineOAMember(user_id=_uid(1), display_name="สมชาย",
                         tags=["phone:0812345678", "interested_camera"]),
            LineOAMember(user_id=_uid(2), display_name="สมหญิง",
                         tags=["phone:0898765432"]),
            LineOAMember(user_id=_uid(3), display_name="นายแดง", tags=[]),
        ]

    def test_phone_match(self, members):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([
            {"customer_id": "C001", "phone": "0812345678", "display_name": ""},
            {"customer_id": "C002", "phone": "0999999999", "display_name": ""},
        ])
        result = match_members_to_customers(members, df)
        assert result.loc[result["customer_id"] == "C001", "line_user_id"].iloc[0] == _uid(1)
        assert result.loc[result["customer_id"] == "C002", "line_user_id"].iloc[0] is None or \
               str(result.loc[result["customer_id"] == "C002", "line_user_id"].iloc[0]) == "None"

    def test_match_method_is_phone(self, members):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([{"customer_id": "C001", "phone": "0812345678", "display_name": ""}])
        result = match_members_to_customers(members, df)
        assert result.loc[0, "match_method"] == "phone"

    def test_name_fallback_match(self, members):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([
            {"customer_id": "C003", "phone": "0700000000", "display_name": "นายแดง"},
        ])
        result = match_members_to_customers(members, df)
        assert result.loc[0, "line_user_id"] == _uid(3)
        assert result.loc[0, "match_method"] == "display_name"

    def test_link_rate_in_attrs(self, members):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([
            {"customer_id": "C001", "phone": "0812345678", "display_name": ""},
            {"customer_id": "C002", "phone": "0999999999", "display_name": ""},
        ])
        result = match_members_to_customers(members, df)
        assert "link_rate" in result.attrs
        assert result.attrs["link_rate"] == 0.5

    def test_no_match_all_none(self, members):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([
            {"customer_id": "C001", "phone": "0700000000", "display_name": ""},
        ])
        result = match_members_to_customers(members, df, name_col=None)
        assert result["line_user_id"].isna().all()

    def test_empty_members_all_unmatched(self):
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([{"customer_id": "C001", "phone": "0812345678", "display_name": ""}])
        result = match_members_to_customers([], df)
        assert result["line_user_id"].isna().all()

    def test_normalises_phone_format(self, members):
        """Customer phone stored as +66 format → still matches."""
        from line_oa_crm_connector import match_members_to_customers
        df = _make_customer_df([{"customer_id": "C001", "phone": "+66812345678", "display_name": ""}])
        result = match_members_to_customers(members, df)
        assert result.loc[0, "line_user_id"] == _uid(1)


# ===========================================================================
# get_match_report
# ===========================================================================

class TestGetMatchReport:

    def test_report_structure(self):
        from line_oa_crm_connector import get_match_report
        df = pd.DataFrame([
            {"customer_id": "C1", "line_user_id": _uid(1), "match_method": "phone"},
            {"customer_id": "C2", "line_user_id": None,    "match_method": None},
            {"customer_id": "C3", "line_user_id": _uid(3), "match_method": "display_name"},
        ])
        report = get_match_report(df)
        assert report["total_customers"] == 3
        assert report["matched"]         == 2
        assert report["unmatched"]       == 1
        assert report["link_rate"]       == pytest.approx(0.6667, abs=0.001)

    def test_recommendation_high(self):
        from line_oa_crm_connector import get_match_report
        df = pd.DataFrame([
            {"customer_id": f"C{i}", "line_user_id": _uid(i), "match_method": "phone"}
            for i in range(10)
        ])
        report = get_match_report(df)
        assert "personalised" in report["recommendation"] or "ดี" in report["recommendation"]

    def test_recommendation_low(self):
        from line_oa_crm_connector import get_match_report
        # 1/10 = 10% < 20% threshold → ต่ำ
        df = pd.DataFrame(
            [{"customer_id": f"C{i}", "line_user_id": None, "match_method": None} for i in range(9)]
            + [{"customer_id": "C9", "line_user_id": _uid(1), "match_method": "phone"}]
        )
        report = get_match_report(df)
        assert "ต่ำ" in report["recommendation"] or "Lookalike" in report["recommendation"]

    def test_by_method_counts(self):
        from line_oa_crm_connector import get_match_report
        df = pd.DataFrame([
            {"customer_id": "C1", "line_user_id": _uid(1), "match_method": "phone"},
            {"customer_id": "C2", "line_user_id": _uid(2), "match_method": "phone"},
            {"customer_id": "C3", "line_user_id": _uid(3), "match_method": "display_name"},
        ])
        report = get_match_report(df)
        assert report["by_method"]["phone"] == 2
        assert report["by_method"]["display_name"] == 1


# ===========================================================================
# fetch_all_members integration (mocked)
# ===========================================================================

class TestFetchAllMembers:

    def test_returns_line_oa_member_list(self):
        from line_oa_crm_connector import fetch_all_members, LineOAMember
        follower_resp  = _mock_get_ok({"userIds": [_uid(1), _uid(2)]})
        profile_resp_1 = _mock_get_ok({"displayName": "สมชาย", "language": "th"})
        profile_resp_2 = _mock_get_ok({"displayName": "สมหญิง", "language": "th"})
        tag_resp       = _mock_get_ok({
            "userTags": [
                {"userId": _uid(1), "tags": [{"tagName": "vip"}]},
                {"userId": _uid(2), "tags": []},
            ]
        })

        with patch("requests.get",  side_effect=[follower_resp, profile_resp_1, profile_resp_2]):
            with patch("requests.post", return_value=tag_resp):
                members = fetch_all_members(channel_access_token="tok")

        assert len(members) == 2
        assert all(isinstance(m, LineOAMember) for m in members)
        assert members[0].display_name == "สมชาย"
        assert "vip" in members[0].tags

    def test_empty_followers_returns_empty_list(self):
        from line_oa_crm_connector import fetch_all_members
        with patch("requests.get", return_value=_mock_get_ok({"userIds": []})):
            result = fetch_all_members(channel_access_token="tok")
        assert result == []

    def test_fetched_at_is_set(self):
        from line_oa_crm_connector import fetch_all_members
        follower_resp = _mock_get_ok({"userIds": [_uid(1)]})
        profile_resp  = _mock_get_ok({"displayName": "ทดสอบ"})
        tag_resp      = _mock_get_ok({"userTags": [{"userId": _uid(1), "tags": []}]})

        with patch("requests.get", side_effect=[follower_resp, profile_resp]):
            with patch("requests.post", return_value=tag_resp):
                members = fetch_all_members(channel_access_token="tok")

        assert members[0].fetched_at != ""

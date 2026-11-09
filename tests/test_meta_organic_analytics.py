"""Tests for meta_organic_analytics.py."""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _fb_post(post_id="123_456", message="Hello FB", eng_rate=0.05, reach=1000):
    return {
        "post_id":        post_id,
        "message":        message,
        "created_time":   "2026-11-01T10:00:00+0000",
        "permalink_url":  f"https://fb.com/{post_id}",
        "platform":       "facebook",
        "impressions":    int(reach * 1.2),
        "reach":          reach,
        "engaged_users":  int(reach * eng_rate),
        "reactions":      10,
        "clicks":         20,
        "video_views":    0,
        "saved":          0,
        "engagement_rate": eng_rate,
        "fetched_at":     "2026-11-08T00:00:00+00:00",
    }


def _ig_post(post_id="ig_001", message="Hello IG", eng_rate=0.08, reach=500):
    return {
        "post_id":        post_id,
        "message":        message,
        "created_time":   "2026-11-02T12:00:00+0000",
        "permalink_url":  f"https://instagram.com/p/{post_id}",
        "media_type":     "IMAGE",
        "like_count":     40,
        "comments_count": 5,
        "platform":       "instagram",
        "impressions":    int(reach * 1.3),
        "reach":          reach,
        "engaged_users":  int(reach * eng_rate),
        "reactions":      0,
        "clicks":         0,
        "video_views":    0,
        "saved":          15,
        "engagement_rate": eng_rate,
        "fetched_at":     "2026-11-08T00:00:00+00:00",
    }


def _make_client(**kwargs):
    from meta_organic_analytics import MetaOrganicClient
    return MetaOrganicClient(
        access_token="fake_token",
        page_id="page_123",
        ig_user_id=kwargs.get("ig_user_id", "ig_456"),
    )


# ===========================================================================
# MetaOrganicClient — init
# ===========================================================================

class TestMetaOrganicClientInit:

    def test_reads_env_vars(self, monkeypatch):
        from meta_organic_analytics import MetaOrganicClient
        monkeypatch.setenv("META_ACCESS_TOKEN",  "tok")
        monkeypatch.setenv("META_PAGE_ID",       "pg")
        monkeypatch.setenv("META_IG_USER_ID",    "ig")
        client = MetaOrganicClient()
        assert client.access_token == "tok"
        assert client.page_id      == "pg"
        assert client.ig_user_id   == "ig"

    def test_explicit_params_override_env(self, monkeypatch):
        from meta_organic_analytics import MetaOrganicClient
        monkeypatch.setenv("META_ACCESS_TOKEN", "env_tok")
        client = MetaOrganicClient(access_token="explicit_tok")
        assert client.access_token == "explicit_tok"

    def test_missing_token_does_not_raise(self, monkeypatch):
        from meta_organic_analytics import MetaOrganicClient
        monkeypatch.delenv("META_ACCESS_TOKEN", raising=False)
        client = MetaOrganicClient()   # should not raise
        assert client.access_token == ""


# ===========================================================================
# fetch_page_posts
# ===========================================================================

class TestFetchPagePosts:

    def _mock_response(self, posts_data, next_url=None):
        resp = MagicMock()
        resp.status_code = 200
        paging = {"next": next_url} if next_url else {}
        resp.json.return_value = {"data": posts_data, "paging": paging}
        return resp

    def test_returns_post_list(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        raw = [
            {"id": "p1", "message": "Post 1", "created_time": "2026-11-01T10:00:00+0000",
             "permalink_url": "https://fb.com/p1"},
        ]
        with patch("requests.get", return_value=self._mock_response(raw)):
            posts = client.fetch_page_posts()
        assert len(posts) == 1
        assert posts[0]["post_id"] == "p1"
        assert posts[0]["platform"] == "facebook"

    def test_message_truncated_to_200(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        raw = [{"id": "p1", "message": "x" * 300, "created_time": "2026-11-01T10:00:00+0000",
                "permalink_url": "https://fb.com/p1"}]
        with patch("requests.get", return_value=self._mock_response(raw)):
            posts = client.fetch_page_posts()
        assert len(posts[0]["message"]) == 200

    def test_since_until_passed_as_params(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_response([])) as mock_get:
            client.fetch_page_posts(since="2026-10-01", until="2026-11-01")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if call_kwargs[1] else call_kwargs[0][1]
        assert params.get("since") == "2026-10-01"
        assert params.get("until") == "2026-11-01"

    def test_pagination_follows_next(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        page1 = [{"id": "p1", "message": "M1", "created_time": "2026-11-01T10:00:00+0000",
                  "permalink_url": "https://fb.com/p1"}]
        page2 = [{"id": "p2", "message": "M2", "created_time": "2026-11-02T10:00:00+0000",
                  "permalink_url": "https://fb.com/p2"}]

        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {
            "data": page1,
            "paging": {"next": "https://graph.facebook.com/v19.0/page_123/posts?after=cursor"},
        }
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"data": page2, "paging": {}}

        with patch("requests.get", side_effect=[resp1, resp2]):
            posts = client.fetch_page_posts()
        assert len(posts) == 2

    def test_empty_response_returns_empty_list(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_response([])):
            assert client.fetch_page_posts() == []


# ===========================================================================
# fetch_post_insights
# ===========================================================================

class TestFetchPostInsights:

    def _mock_insights(self, impressions=1200, reach=1000, engaged=50,
                       reactions={"like": 20}, clicks=30, video_views=0):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {"name": "post_impressions", "values": [{"value": impressions}]},
                {"name": "post_reach",        "values": [{"value": reach}]},
                {"name": "post_engaged_users","values": [{"value": engaged}]},
                {"name": "post_reactions_by_type_total", "values": [{"value": reactions}]},
                {"name": "post_clicks",       "values": [{"value": clicks}]},
                {"name": "post_video_views",  "values": [{"value": video_views}]},
            ]
        }
        return resp

    def test_returns_all_keys(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_insights()):
            result = client.fetch_post_insights("p1")
        for key in ("impressions", "reach", "engaged_users", "reactions",
                    "clicks", "video_views", "engagement_rate"):
            assert key in result

    def test_engagement_rate_calculation(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_insights(reach=1000, engaged=100)):
            result = client.fetch_post_insights("p1")
        assert result["engagement_rate"] == pytest.approx(0.10, abs=0.001)

    def test_zero_reach_no_division_error(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_insights(reach=0, engaged=0)):
            result = client.fetch_post_insights("p1")
        assert result["engagement_rate"] == 0.0

    def test_reactions_summed_from_dict(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        reactions = {"like": 20, "love": 5, "haha": 3}
        with patch("requests.get", return_value=self._mock_insights(reactions=reactions)):
            result = client.fetch_post_insights("p1")
        assert result["reactions"] == 28


# ===========================================================================
# fetch_ig_media
# ===========================================================================

class TestFetchIgMedia:

    def _mock_ig_response(self, media_data, next_url=None):
        resp = MagicMock()
        resp.status_code = 200
        paging = {"next": next_url} if next_url else {}
        resp.json.return_value = {"data": media_data, "paging": paging}
        return resp

    def test_returns_ig_posts(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        raw = [{"id": "ig1", "caption": "IG post", "media_type": "IMAGE",
                "timestamp": "2026-11-01T10:00:00+0000",
                "like_count": 50, "comments_count": 5,
                "permalink": "https://instagram.com/p/ig1"}]
        with patch("requests.get", return_value=self._mock_ig_response(raw)):
            posts = client.fetch_ig_media()
        assert len(posts) == 1
        assert posts[0]["platform"] == "instagram"
        assert posts[0]["post_id"] == "ig1"

    def test_no_ig_user_id_returns_empty(self):
        from meta_organic_analytics import MetaOrganicClient
        client = MetaOrganicClient(access_token="tok", page_id="pg", ig_user_id="")
        with patch("requests.get") as mock_get:
            posts = client.fetch_ig_media()
        mock_get.assert_not_called()
        assert posts == []

    def test_caption_truncated_to_200(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        raw = [{"id": "ig1", "caption": "y" * 300, "media_type": "IMAGE",
                "timestamp": "2026-11-01T10:00:00+0000",
                "like_count": 10, "comments_count": 1,
                "permalink": "https://instagram.com/p/ig1"}]
        with patch("requests.get", return_value=self._mock_ig_response(raw)):
            posts = client.fetch_ig_media()
        assert len(posts[0]["message"]) == 200


# ===========================================================================
# fetch_ig_media_insights
# ===========================================================================

class TestFetchIgMediaInsights:

    def _mock_ig_insights(self, impressions=600, reach=500, engagement=40, saved=10):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {"name": "impressions", "values": [{"value": impressions}]},
                {"name": "reach",       "values": [{"value": reach}]},
                {"name": "engagement",  "values": [{"value": engagement}]},
                {"name": "saved",       "values": [{"value": saved}]},
            ]
        }
        return resp

    def test_returns_all_keys(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_ig_insights()):
            result = client.fetch_ig_media_insights("ig1")
        for key in ("impressions", "reach", "engaged_users", "saved", "engagement_rate"):
            assert key in result

    def test_engagement_rate_ig(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_ig_insights(reach=500, engagement=50)):
            result = client.fetch_ig_media_insights("ig1")
        assert result["engagement_rate"] == pytest.approx(0.10, abs=0.001)

    def test_saved_count_included(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        with patch("requests.get", return_value=self._mock_ig_insights(saved=25)):
            result = client.fetch_ig_media_insights("ig1")
        assert result["saved"] == 25


# ===========================================================================
# fetch_all_posts_with_insights
# ===========================================================================

class TestFetchAllPostsWithInsights:

    def test_combines_fb_and_ig(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()

        fb_post_raw = [{"id": "fb1", "message": "FB", "created_time": "2026-11-01T10:00:00+0000",
                        "permalink_url": "https://fb.com/fb1"}]
        ig_media_raw = [{"id": "ig1", "caption": "IG", "media_type": "IMAGE",
                         "timestamp": "2026-11-01T12:00:00+0000",
                         "like_count": 10, "comments_count": 2,
                         "permalink": "https://instagram.com/p/ig1"}]
        insights = {"impressions": 1000, "reach": 800, "engaged_users": 50,
                    "reactions": 5, "clicks": 10, "video_views": 0, "saved": 0,
                    "engagement_rate": 0.0625}

        with patch.object(client, "fetch_page_posts", return_value=[
            {"post_id": "fb1", "message": "FB", "created_time": "2026-11-01T10:00:00+0000",
             "permalink_url": "https://fb.com/fb1", "platform": "facebook"}
        ]), patch.object(client, "fetch_ig_media", return_value=[
            {"post_id": "ig1", "message": "IG", "created_time": "2026-11-01T12:00:00+0000",
             "permalink_url": "https://instagram.com/p/ig1", "platform": "instagram",
             "media_type": "IMAGE", "like_count": 10, "comments_count": 2}
        ]), patch.object(client, "fetch_post_insights", return_value=insights), \
           patch.object(client, "fetch_ig_media_insights", return_value=insights):
            posts = client.fetch_all_posts_with_insights(days=30)

        assert len(posts) == 2
        platforms = {p["platform"] for p in posts}
        assert platforms == {"facebook", "instagram"}

    def test_insight_failure_does_not_drop_post(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()

        with patch.object(client, "fetch_page_posts", return_value=[
            {"post_id": "fb1", "message": "FB", "created_time": "2026-11-01T10:00:00+0000",
             "permalink_url": "https://fb.com/fb1", "platform": "facebook"}
        ]), patch.object(client, "fetch_ig_media", return_value=[]), \
           patch.object(client, "fetch_post_insights", side_effect=RuntimeError("API down")):
            posts = client.fetch_all_posts_with_insights()

        assert len(posts) == 1
        assert posts[0]["impressions"] == 0   # empty insights fallback

    def test_fetched_at_field_set(self):
        from meta_organic_analytics import MetaOrganicClient
        client = _make_client()
        insights = {"impressions": 100, "reach": 80, "engaged_users": 5,
                    "reactions": 1, "clicks": 2, "video_views": 0, "saved": 0,
                    "engagement_rate": 0.0625}
        with patch.object(client, "fetch_page_posts", return_value=[
            {"post_id": "fb1", "message": "FB", "created_time": "2026-11-01T10:00:00+0000",
             "permalink_url": "https://fb.com/fb1", "platform": "facebook"}
        ]), patch.object(client, "fetch_ig_media", return_value=[]), \
           patch.object(client, "fetch_post_insights", return_value=insights):
            posts = client.fetch_all_posts_with_insights()
        assert "fetched_at" in posts[0]


# ===========================================================================
# save_organic_analytics / load_organic_analytics
# ===========================================================================

class TestSaveLoadOrganicAnalytics:

    def test_save_writes_jsonl(self, tmp_path):
        from meta_organic_analytics import save_organic_analytics, load_organic_analytics
        posts = [_fb_post("p1"), _ig_post("ig1")]
        save_organic_analytics(str(tmp_path), posts)
        loaded = load_organic_analytics(str(tmp_path))
        assert len(loaded) == 2

    def test_save_appends_on_second_call(self, tmp_path):
        from meta_organic_analytics import save_organic_analytics, load_organic_analytics
        save_organic_analytics(str(tmp_path), [_fb_post("p1")])
        save_organic_analytics(str(tmp_path), [_fb_post("p2")])
        loaded = load_organic_analytics(str(tmp_path))
        assert len(loaded) == 2

    def test_dry_run_does_not_write(self, tmp_path):
        from meta_organic_analytics import save_organic_analytics
        save_organic_analytics(str(tmp_path), [_fb_post("p1")], dry_run=True)
        assert not os.path.exists(str(tmp_path / "organic_analytics.jsonl"))

    def test_load_missing_file_returns_empty(self, tmp_path):
        from meta_organic_analytics import load_organic_analytics
        assert load_organic_analytics(str(tmp_path)) == []

    def test_load_skips_malformed_lines(self, tmp_path):
        from meta_organic_analytics import load_organic_analytics
        path = str(tmp_path / "organic_analytics.jsonl")
        with open(path, "w") as f:
            f.write('{"post_id": "p1"}\n')
            f.write("not_json\n")
            f.write('{"post_id": "p2"}\n')
        loaded = load_organic_analytics(str(tmp_path))
        assert len(loaded) == 2


# ===========================================================================
# get_top_posts
# ===========================================================================

class TestGetTopPosts:

    def test_returns_top_n_sorted_by_engagement_rate(self):
        from meta_organic_analytics import get_top_posts
        posts = [
            _fb_post("p1", eng_rate=0.03, reach=100),
            _fb_post("p2", eng_rate=0.10, reach=200),
            _fb_post("p3", eng_rate=0.07, reach=150),
        ]
        top = get_top_posts(posts, top_n=2)
        assert top[0]["post_id"] == "p2"
        assert top[1]["post_id"] == "p3"

    def test_min_reach_filter(self):
        from meta_organic_analytics import get_top_posts
        posts = [
            _fb_post("p1", eng_rate=0.50, reach=10),   # high eng but low reach
            _fb_post("p2", eng_rate=0.08, reach=500),
        ]
        top = get_top_posts(posts, top_n=5, min_reach=50)
        assert all(p["reach"] >= 50 for p in top)
        assert len(top) == 1

    def test_platform_filter(self):
        from meta_organic_analytics import get_top_posts
        posts = [_fb_post("p1"), _ig_post("ig1")]
        fb_only = get_top_posts(posts, platform="facebook")
        assert all(p["platform"] == "facebook" for p in fb_only)

    def test_rank_by_reach(self):
        from meta_organic_analytics import get_top_posts
        posts = [_fb_post("p1", reach=100), _fb_post("p2", reach=500)]
        top = get_top_posts(posts, rank_by="reach", min_reach=1)
        assert top[0]["post_id"] == "p2"

    def test_empty_posts_returns_empty(self):
        from meta_organic_analytics import get_top_posts
        assert get_top_posts([]) == []


# ===========================================================================
# get_organic_report
# ===========================================================================

class TestGetOrganicReport:

    def test_contains_post_counts(self):
        from meta_organic_analytics import get_organic_report
        posts = [_fb_post("p1"), _ig_post("ig1")]
        report = get_organic_report(posts)
        assert "2" in report   # 2 posts total
        assert "FB" in report
        assert "IG" in report

    def test_contains_reach_and_engagement(self):
        from meta_organic_analytics import get_organic_report
        posts = [_fb_post("p1", reach=1000, eng_rate=0.05)]
        report = get_organic_report(posts)
        assert "1,000" in report or "1000" in report
        assert "5.00" in report or "5.0" in report   # 5% engagement

    def test_month_label_in_report(self):
        from meta_organic_analytics import get_organic_report
        posts = [_fb_post("p1")]
        report = get_organic_report(posts, month="2026-11")
        assert "2026-11" in report

    def test_empty_posts_returns_no_data_message(self):
        from meta_organic_analytics import get_organic_report
        report = get_organic_report([])
        assert "ไม่มีข้อมูล" in report

    def test_thai_header(self):
        from meta_organic_analytics import get_organic_report
        posts = [_fb_post("p1")]
        report = get_organic_report(posts)
        assert "รายงาน" in report

    def test_top_posts_section_included(self):
        from meta_organic_analytics import get_organic_report
        posts = [_fb_post(f"p{i}", reach=100 * (i + 1), eng_rate=0.05) for i in range(6)]
        report = get_organic_report(posts)
        assert "Top posts" in report

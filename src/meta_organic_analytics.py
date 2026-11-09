"""meta_organic_analytics.py — Pull and analyse organic Facebook/Instagram post metrics.

Separates organic post performance from paid ad performance so stores can see
which content drives engagement without ad spend.

Meta Graph API endpoints used
------------------------------
Facebook (Page posts):
    GET /{page_id}/posts            → list of posts
    GET /{post_id}/insights         → impressions, reach, engaged_users, reactions, clicks

Instagram (Business Account media):
    GET /{ig_user_id}/media         → list of media
    GET /{media_id}/insights        → impressions, reach, engagement, saved

Authentication
--------------
Set env vars:
    META_ACCESS_TOKEN  — permanent page/system-user token
    META_PAGE_ID       — Facebook Page ID
    META_IG_USER_ID    — Instagram Business Account ID (optional — skipped if absent)

Storage
-------
    data/organic_analytics.jsonl  — one record per post/media (append per fetch)

Usage
-----
    from meta_organic_analytics import MetaOrganicClient, save_organic_analytics

    client = MetaOrganicClient()
    posts  = client.fetch_all_posts_with_insights(days=30)
    save_organic_analytics("data", posts)
    print(get_organic_report(posts))
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

_logger = logging.getLogger(__name__)

_META_GRAPH_URL  = "https://graph.facebook.com/v19.0"
_MAX_RETRIES     = 3
_RETRY_DELAY     = 2.0
_REQUEST_TIMEOUT = 30

# Facebook post insight metrics requested
_FB_POST_INSIGHTS = ",".join([
    "post_impressions",
    "post_reach",
    "post_engaged_users",
    "post_reactions_by_type_total",
    "post_clicks",
    "post_video_views",
])

# Instagram media insight metrics requested
_IG_MEDIA_INSIGHTS = "impressions,reach,engagement,saved"

# Instagram media fields (non-insight data returned by /media)
_IG_MEDIA_FIELDS = "id,caption,media_type,timestamp,like_count,comments_count,permalink"


# ---------------------------------------------------------------------------
# MetaOrganicClient
# ---------------------------------------------------------------------------

@dataclass
class MetaOrganicClient:
    """Client for fetching organic Page and Instagram post analytics.

    Parameters
    ----------
    access_token : str, optional
        Falls back to env ``META_ACCESS_TOKEN``.
    page_id : str, optional
        Facebook Page ID. Falls back to ``META_PAGE_ID``.
    ig_user_id : str, optional
        Instagram Business Account ID. Falls back to ``META_IG_USER_ID``.
        If absent, Instagram fetches are skipped.
    """
    access_token: str | None = None
    page_id:      str | None = None
    ig_user_id:   str | None = None

    def __post_init__(self) -> None:
        self.access_token = self.access_token or os.environ.get("META_ACCESS_TOKEN", "")
        self.page_id      = self.page_id      or os.environ.get("META_PAGE_ID",      "")
        self.ig_user_id   = self.ig_user_id   or os.environ.get("META_IG_USER_ID",   "")

        if not self.access_token:
            _logger.warning("META_ACCESS_TOKEN not set — API calls will fail")
        if not self.page_id:
            _logger.warning("META_PAGE_ID not set — Facebook post fetches will fail")

    # ------------------------------------------------------------------
    # Facebook Page posts
    # ------------------------------------------------------------------

    def fetch_page_posts(
        self,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return a list of published Page posts (basic fields only, no insights).

        Parameters
        ----------
        since : str, optional
            ISO-8601 date string lower bound e.g. "2026-10-01".
        until : str, optional
            ISO-8601 date string upper bound e.g. "2026-11-01".
        limit : int
            Max posts per page (Meta max = 100).

        Returns
        -------
        list[dict]
            Each dict has: ``{post_id, message, created_time, permalink_url, platform}``
        """
        params: dict[str, Any] = {
            "fields": "id,message,created_time,permalink_url",
            "limit":  limit,
        }
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        posts: list[dict] = []
        endpoint = f"{self.page_id}/posts"

        while endpoint:
            data = self._get(endpoint, params=params)
            for raw in data.get("data", []):
                posts.append({
                    "post_id":       raw.get("id", ""),
                    "message":       (raw.get("message", "") or "")[:200],
                    "created_time":  raw.get("created_time", ""),
                    "permalink_url": raw.get("permalink_url", ""),
                    "platform":      "facebook",
                })
            # Cursor pagination
            next_url = data.get("paging", {}).get("next")
            if next_url:
                # Strip the base URL — _get adds it back
                endpoint = next_url.replace(f"{_META_GRAPH_URL}/", "")
                params = {}  # params are already embedded in next_url
            else:
                endpoint = None  # type: ignore[assignment]

        _logger.info("fetch_page_posts: retrieved %d posts", len(posts))
        return posts

    def fetch_post_insights(self, post_id: str) -> dict:
        """Return engagement insights for a single Facebook post.

        Returns
        -------
        dict
            ``{impressions, reach, engaged_users, reactions, clicks,
               video_views, engagement_rate}``
        """
        data = self._get(f"{post_id}/insights", params={"metric": _FB_POST_INSIGHTS})
        metrics: dict[str, Any] = {}
        for item in data.get("data", []):
            name  = item.get("name", "")
            value = item.get("values", [{}])[-1].get("value", 0) if item.get("values") else 0
            metrics[name] = value

        impressions   = int(metrics.get("post_impressions", 0) or 0)
        reach         = int(metrics.get("post_reach", 0) or 0)
        engaged_users = int(metrics.get("post_engaged_users", 0) or 0)
        clicks        = int(metrics.get("post_clicks", 0) or 0)
        video_views   = int(metrics.get("post_video_views", 0) or 0)
        reactions_raw = metrics.get("post_reactions_by_type_total", {})
        reactions     = sum(reactions_raw.values()) if isinstance(reactions_raw, dict) else int(reactions_raw or 0)

        engagement_rate = round(engaged_users / reach, 4) if reach > 0 else 0.0

        return {
            "impressions":    impressions,
            "reach":          reach,
            "engaged_users":  engaged_users,
            "reactions":      reactions,
            "clicks":         clicks,
            "video_views":    video_views,
            "engagement_rate": engagement_rate,
        }

    # ------------------------------------------------------------------
    # Instagram media
    # ------------------------------------------------------------------

    def fetch_ig_media(
        self,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return a list of Instagram media objects (basic fields only).

        Requires ``ig_user_id`` to be set.  Returns ``[]`` if absent.

        Returns
        -------
        list[dict]
            Each dict has: ``{post_id, message, created_time, permalink_url,
                              media_type, like_count, comments_count, platform}``
        """
        if not self.ig_user_id:
            _logger.debug("fetch_ig_media: ig_user_id not set — skipping")
            return []

        params: dict[str, Any] = {
            "fields": _IG_MEDIA_FIELDS,
            "limit":  limit,
        }
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        posts: list[dict] = []
        endpoint = f"{self.ig_user_id}/media"

        while endpoint:
            data = self._get(endpoint, params=params)
            for raw in data.get("data", []):
                posts.append({
                    "post_id":       raw.get("id", ""),
                    "message":       (raw.get("caption", "") or "")[:200],
                    "created_time":  raw.get("timestamp", ""),
                    "permalink_url": raw.get("permalink", ""),
                    "media_type":    raw.get("media_type", ""),
                    "like_count":    int(raw.get("like_count", 0) or 0),
                    "comments_count": int(raw.get("comments_count", 0) or 0),
                    "platform":      "instagram",
                })
            next_url = data.get("paging", {}).get("next")
            if next_url:
                endpoint = next_url.replace(f"{_META_GRAPH_URL}/", "")
                params = {}
            else:
                endpoint = None  # type: ignore[assignment]

        _logger.info("fetch_ig_media: retrieved %d posts", len(posts))
        return posts

    def fetch_ig_media_insights(self, media_id: str) -> dict:
        """Return engagement insights for a single Instagram media object.

        Returns
        -------
        dict
            ``{impressions, reach, engagement, saved, engagement_rate}``
        """
        data = self._get(f"{media_id}/insights", params={"metric": _IG_MEDIA_INSIGHTS})
        metrics: dict[str, int] = {}
        for item in data.get("data", []):
            metrics[item.get("name", "")] = int(item.get("values", [{}])[0].get("value", 0)
                                                 if item.get("values")
                                                 else item.get("value", 0) or 0)

        impressions = metrics.get("impressions", 0)
        reach       = metrics.get("reach",       0)
        engagement  = metrics.get("engagement",  0)
        saved       = metrics.get("saved",       0)

        engagement_rate = round(engagement / reach, 4) if reach > 0 else 0.0

        return {
            "impressions":    impressions,
            "reach":          reach,
            "engaged_users":  engagement,
            "reactions":      0,          # IG doesn't expose reactions separately
            "clicks":         0,
            "video_views":    0,
            "saved":          saved,
            "engagement_rate": engagement_rate,
        }

    # ------------------------------------------------------------------
    # Combined: posts + insights in one call
    # ------------------------------------------------------------------

    def fetch_all_posts_with_insights(
        self,
        days: int = 30,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        """Fetch FB + IG posts and their insights for a time range.

        Parameters
        ----------
        days : int
            Look-back window if ``since``/``until`` not provided.
        since, until : str, optional
            Override look-back window with explicit ISO-8601 dates.

        Returns
        -------
        list[dict]
            Combined FB + IG posts, each enriched with insight metrics.
            Keys: ``{post_id, message, created_time, permalink_url, platform,
                      impressions, reach, engaged_users, reactions, clicks,
                      video_views, saved, engagement_rate, fetched_at}``
        """
        if not since:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        if not until:
            until = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        fb_posts = self.fetch_page_posts(since=since, until=until)
        ig_posts = self.fetch_ig_media(since=since, until=until)

        combined: list[dict] = []
        fetched_at = datetime.now(timezone.utc).isoformat()

        for post in fb_posts:
            try:
                insights = self.fetch_post_insights(post["post_id"])
            except Exception as exc:
                _logger.warning("fetch_all_posts_with_insights: insights failed for %s — %s",
                                post["post_id"], exc)
                insights = _empty_insights()
            combined.append({**post, **insights, "fetched_at": fetched_at,
                              "saved": insights.get("saved", 0)})

        for post in ig_posts:
            try:
                insights = self.fetch_ig_media_insights(post["post_id"])
            except Exception as exc:
                _logger.warning("fetch_all_posts_with_insights: IG insights failed for %s — %s",
                                post["post_id"], exc)
                insights = _empty_insights()
            combined.append({**post, **insights, "fetched_at": fetched_at})

        _logger.info("fetch_all_posts_with_insights: %d FB + %d IG = %d total",
                     len(fb_posts), len(ig_posts), len(combined))
        return combined

    # ------------------------------------------------------------------
    # HTTP helpers (same pattern as MetaAdsClient)
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        return self._request("GET", endpoint, params=params)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
    ) -> dict:
        try:
            import requests as _requests
        except ImportError:
            _logger.error("'requests' package not installed")
            return {}

        # Support both short endpoint names and full next-page URLs
        if endpoint.startswith("http"):
            url = endpoint
            base_params = {}
        else:
            url = f"{_META_GRAPH_URL}/{endpoint}"
            base_params = {"access_token": self.access_token}

        if params:
            base_params.update(params)

        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = _requests.get(url, params=base_params, timeout=_REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    delay = _RETRY_DELAY * (2 ** (attempt - 1))
                    _logger.warning("Meta API: 429 rate-limit — retry %d/%d in %.1fs",
                                    attempt, _MAX_RETRIES, delay)
                    time.sleep(delay)
                    continue

                _logger.error("Meta API: HTTP %d on GET %s — %s",
                              resp.status_code, endpoint, resp.text[:200])
                resp.raise_for_status()

            except Exception as exc:
                last_exc = exc
                delay = _RETRY_DELAY * (2 ** (attempt - 1))
                _logger.warning("Meta API: %s on attempt %d/%d — retry in %.1fs",
                                type(exc).__name__, attempt, _MAX_RETRIES, delay)
                if attempt < _MAX_RETRIES:
                    time.sleep(delay)

        raise RuntimeError(
            f"Meta API: all {_MAX_RETRIES} retries exhausted for GET {endpoint}. "
            f"Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_organic_analytics(
    data_dir: str,
    posts: list[dict],
    dry_run: bool = False,
) -> str:
    """Append posts to organic_analytics.jsonl.

    Returns
    -------
    str
        Path to the log file (even if dry_run=True).
    """
    path = os.path.join(data_dir, "organic_analytics.jsonl")
    if not dry_run and posts:
        os.makedirs(data_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + "\n")
        _logger.info("save_organic_analytics: wrote %d records to %s", len(posts), path)
    return path


def load_organic_analytics(data_dir: str) -> list[dict]:
    """Load all records from organic_analytics.jsonl.  Returns [] if file absent."""
    path = os.path.join(data_dir, "organic_analytics.jsonl")
    if not os.path.exists(path):
        return []
    records: list[dict] = []
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


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def get_top_posts(
    posts: list[dict],
    top_n: int = 5,
    rank_by: str = "engagement_rate",
    min_reach: int = 50,
    platform: str = "",
) -> list[dict]:
    """Return top-performing posts sorted by a metric.

    Parameters
    ----------
    posts : list[dict]
        Output of ``fetch_all_posts_with_insights`` or ``load_organic_analytics``.
    top_n : int
    rank_by : str
        One of: ``engagement_rate``, ``reach``, ``impressions``, ``clicks``, ``saved``.
    min_reach : int
        Minimum reach to qualify (filters out very low signal posts).
    platform : str
        Filter to "facebook" or "instagram". Empty = both.

    Returns
    -------
    list[dict]
        Top posts sorted descending.
    """
    filtered = [
        p for p in posts
        if p.get("reach", 0) >= min_reach
        and (not platform or p.get("platform", "") == platform)
    ]
    return sorted(filtered, key=lambda p: p.get(rank_by, 0), reverse=True)[:top_n]


def get_organic_report(
    posts: list[dict],
    month: str = "",
) -> str:
    """Return a Thai summary of organic post performance.

    Parameters
    ----------
    posts : list[dict]
        All posts (FB + IG) for the period.
    month : str
        Label for the report header e.g. "2026-11".

    Returns
    -------
    str
        Multi-line Thai summary.
    """
    if not posts:
        return "ไม่มีข้อมูล organic posts ในช่วงเวลานี้"

    month_label = f" ({month})" if month else ""
    fb_posts = [p for p in posts if p.get("platform") == "facebook"]
    ig_posts = [p for p in posts if p.get("platform") == "instagram"]

    total_reach      = sum(p.get("reach", 0) for p in posts)
    total_impressions = sum(p.get("impressions", 0) for p in posts)
    total_engaged    = sum(p.get("engaged_users", 0) for p in posts)
    avg_eng_rate     = (
        round(sum(p.get("engagement_rate", 0) for p in posts) / len(posts), 4)
        if posts else 0.0
    )

    top5 = get_top_posts(posts, top_n=5, rank_by="engagement_rate", min_reach=1)

    lines = [
        f"รายงาน Organic Analytics{month_label}",
        f"  โพสต์ทั้งหมด           : {len(posts):,} โพสต์  "
        f"(FB {len(fb_posts):,} / IG {len(ig_posts):,})",
        f"  Reach รวม              : {total_reach:,}",
        f"  Impressions รวม        : {total_impressions:,}",
        f"  Engaged users รวม      : {total_engaged:,}",
        f"  Avg engagement rate    : {avg_eng_rate*100:.2f}%",
    ]

    if top5:
        lines.append("")
        lines.append("  Top posts (engagement rate):")
        for i, p in enumerate(top5, 1):
            platform_tag = "FB" if p.get("platform") == "facebook" else "IG"
            snippet = (p.get("message", "") or "")[:50].replace("\n", " ")
            lines.append(
                f"    {i}. [{platform_tag}] {snippet!r:<55} "
                f"eng={p.get('engagement_rate',0)*100:.1f}%  reach={p.get('reach',0):,}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_insights() -> dict:
    return {
        "impressions":    0,
        "reach":          0,
        "engaged_users":  0,
        "reactions":      0,
        "clicks":         0,
        "video_views":    0,
        "saved":          0,
        "engagement_rate": 0.0,
    }

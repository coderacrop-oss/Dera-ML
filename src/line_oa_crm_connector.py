"""line_oa_crm_connector.py — Pull member data + interest tags from LINE OA.

Fetches the full follower list from LINE Messaging API, enriches each member
with profile data and OA Manager tags, then matches LINE userId back to
customer records in POS/CRM data via phone number or display name.

LINE OA API endpoints used
--------------------------
  GET /v2/bot/followers          — paginated userId list
  GET /v2/bot/profile/{userId}   — displayName, pictureUrl, language
  POST /v2/bot/users/tag/list    — batch tag fetch (up to 100 userIds)

Authentication
--------------
Set env var ``LINE_CHANNEL_ACCESS_TOKEN`` or pass ``channel_access_token``
to each function.

Usage
-----
    from line_oa_crm_connector import fetch_all_members, match_members_to_customers
    import pandas as pd

    members_df = fetch_all_members(channel_access_token="...")
    # columns: user_id, display_name, language, tags, fetched_at

    pos_df = pd.read_csv("pos.csv")  # must have customer_id + phone columns
    crm_df = pd.read_csv("crm.csv")  # must have customer_id + phone columns

    matched = match_members_to_customers(members_df, crm_df)
    # adds line_user_id column to crm_df rows that matched
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

_LINE_API_BASE      = "https://api.line.me/v2/bot"
_PROFILE_BATCH_SIZE = 100   # /users/tag/list accepts up to 100 userIds
_MAX_RETRIES        = 3
_RETRY_BASE_DELAY   = 1.0
_REQUEST_TIMEOUT    = 20
_FOLLOWERS_LIMIT    = 1000  # LINE API max per page


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineOAMember:
    """One LINE OA follower with profile + tags."""
    user_id:      str
    display_name: str    = ""
    picture_url:  str    = ""
    language:     str    = ""
    tags:         list[str] = field(default_factory=list)
    fetched_at:   str    = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_followers(
    channel_access_token: str | None = None,
    limit: int = _FOLLOWERS_LIMIT,
) -> list[str]:
    """Return all LINE userId strings that follow the OA.

    Uses cursor-based pagination — fetches until no next cursor.

    Parameters
    ----------
    channel_access_token : str, optional
        Falls back to env ``LINE_CHANNEL_ACCESS_TOKEN``.
    limit : int
        Page size per request (max 1000, LINE default).

    Returns
    -------
    list[str]
        All follower userIds.
    """
    token = _resolve_token(channel_access_token)
    all_ids: list[str] = []
    next_token: str | None = None

    while True:
        params: dict = {"limit": limit}
        if next_token:
            params["start"] = next_token

        resp = _get(f"{_LINE_API_BASE}/followers", params=params, token=token)
        user_ids = resp.get("userIds", [])
        all_ids.extend(user_ids)

        next_token = resp.get("next")
        if not next_token:
            break

    _logger.info("fetch_followers: retrieved %d userIds", len(all_ids))
    return all_ids


def get_member_profile(
    user_id: str,
    channel_access_token: str | None = None,
) -> dict:
    """Fetch profile for a single userId.

    Returns
    -------
    dict
        ``{displayName, pictureUrl, language}``
        Empty dict if user not found or blocked.
    """
    token = _resolve_token(channel_access_token)
    try:
        return _get(f"{_LINE_API_BASE}/profile/{user_id}", token=token)
    except Exception as exc:
        _logger.warning("get_member_profile(%s): %s", user_id, exc)
        return {}


def get_member_tags(
    user_ids: list[str],
    channel_access_token: str | None = None,
) -> dict[str, list[str]]:
    """Fetch OA Manager tags for a batch of userIds (max 100 per call).

    Returns
    -------
    dict[str, list[str]]
        ``{userId: [tag1, tag2, ...]}``
    """
    token   = _resolve_token(channel_access_token)
    result: dict[str, list[str]] = {uid: [] for uid in user_ids}

    for i in range(0, len(user_ids), _PROFILE_BATCH_SIZE):
        batch = user_ids[i: i + _PROFILE_BATCH_SIZE]
        try:
            resp = _post(
                f"{_LINE_API_BASE}/users/tag/list",
                body={"userIds": batch},
                token=token,
            )
            for entry in resp.get("userTags", []):
                uid  = entry.get("userId", "")
                tags = [t.get("tagName", "") for t in entry.get("tags", [])]
                result[uid] = [t for t in tags if t]
        except Exception as exc:
            _logger.warning("get_member_tags batch %d: %s", i, exc)

    return result


def fetch_all_members(
    channel_access_token: str | None = None,
    include_tags: bool = True,
) -> list[LineOAMember]:
    """Fetch full member list: userId + profile + tags.

    Parameters
    ----------
    channel_access_token : str, optional
    include_tags : bool
        If True, also fetch OA Manager tags per member.
        Set False to skip tag fetch when not needed.

    Returns
    -------
    list[LineOAMember]
    """
    token      = _resolve_token(channel_access_token)
    now_iso    = datetime.now(timezone.utc).isoformat()
    user_ids   = fetch_followers(channel_access_token=token)

    if not user_ids:
        return []

    # Fetch tags in batch
    tags_map: dict[str, list[str]] = {}
    if include_tags:
        tags_map = get_member_tags(user_ids, channel_access_token=token)

    # Fetch profiles — batched one-by-one (LINE has no batch profile endpoint)
    members: list[LineOAMember] = []
    for uid in user_ids:
        profile = get_member_profile(uid, channel_access_token=token)
        members.append(LineOAMember(
            user_id      = uid,
            display_name = profile.get("displayName", ""),
            picture_url  = profile.get("pictureUrl",  ""),
            language     = profile.get("language",    ""),
            tags         = tags_map.get(uid, []),
            fetched_at   = now_iso,
        ))

    _logger.info("fetch_all_members: fetched %d members (tags=%s)", len(members), include_tags)
    return members


def match_members_to_customers(
    members: list[LineOAMember],
    customer_df,
    phone_col: str = "phone",
    customer_id_col: str = "customer_id",
    name_col: str | None = "display_name",
) -> Any:
    """Match LINE members to customer records via phone number.

    Strategy (in order):
      1. Exact phone match (normalised to +66 format)
      2. Display name match (fallback, lower confidence)

    Adds column ``line_user_id`` to customer_df rows that matched.
    Unmatched rows get ``line_user_id = None``.

    Parameters
    ----------
    members : list[LineOAMember]
    customer_df : pd.DataFrame
        Must contain at minimum ``customer_id`` and ``phone`` columns.
    phone_col : str
        Column name for phone in customer_df.
    customer_id_col : str
        Column name for customer ID in customer_df.
    name_col : str | None
        Column name for customer display name (used as fallback).
        Pass None to skip name matching.

    Returns
    -------
    pd.DataFrame
        customer_df with added ``line_user_id`` and ``match_method`` columns.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for match_members_to_customers")

    df = customer_df.copy()
    df["line_user_id"] = None
    df["match_method"] = None

    # Build phone → userId lookup
    phone_map: dict[str, str] = {}
    name_map:  dict[str, str] = {}

    for m in members:
        # Tags may contain phone numbers ("phone:0812345678")
        for tag in m.tags:
            if tag.startswith("phone:"):
                raw = tag.split(":", 1)[1]
                phone_map[_normalise_phone(raw)] = m.user_id

        if m.display_name:
            name_map[m.display_name.strip().lower()] = m.user_id

    # Apply phone match
    if phone_col in df.columns:
        def _match_phone(row):
            if pd.notna(row.get("line_user_id")):
                return row
            norm = _normalise_phone(str(row.get(phone_col, "") or ""))
            if norm and norm in phone_map:
                row["line_user_id"] = phone_map[norm]
                row["match_method"] = "phone"
            return row

        df = df.apply(_match_phone, axis=1)

    # Fallback: display name match
    if name_col and name_col in df.columns:
        def _match_name(row):
            if pd.notna(row.get("line_user_id")):
                return row
            name = str(row.get(name_col, "") or "").strip().lower()
            if name and name in name_map:
                row["line_user_id"] = name_map[name]
                row["match_method"] = "display_name"
            return row

        df = df.apply(_match_name, axis=1)

    matched   = df["line_user_id"].notna().sum()
    total     = len(df)
    link_rate = round(matched / total, 4) if total > 0 else 0.0

    _logger.info(
        "match_members_to_customers: %d/%d matched (%.1f%%)",
        matched, total, link_rate * 100,
    )
    df.attrs["link_rate"] = link_rate
    return df


def get_match_report(matched_df) -> dict:
    """Return a summary report of the match quality.

    Parameters
    ----------
    matched_df : pd.DataFrame
        Output of match_members_to_customers().

    Returns
    -------
    dict
        ``{total_customers, matched, unmatched, link_rate, by_method, recommendation}``
    """
    total   = len(matched_df)
    matched = int(matched_df["line_user_id"].notna().sum())
    by_method: dict[str, int] = {}

    if "match_method" in matched_df.columns:
        counts = matched_df["match_method"].value_counts().to_dict()
        by_method = {str(k): int(v) for k, v in counts.items()}

    link_rate = round(matched / total, 4) if total > 0 else 0.0

    if link_rate >= 0.50:
        recommendation = "คุณภาพดี — ส่ง broadcast แบบ personalised ได้"
    elif link_rate >= 0.20:
        recommendation = "คุณภาพปานกลาง — แนะนำให้เพิ่ม phone tag ใน LINE CRM ให้ครบกว่านี้"
    else:
        recommendation = "คุณภาพต่ำ — ให้ทีมขายขอ LINE ID จากลูกค้าโดยตรง หรือใช้ Lookalike แทน"

    return {
        "total_customers": total,
        "matched":         matched,
        "unmatched":       total - matched,
        "link_rate":       link_rate,
        "by_method":       by_method,
        "recommendation":  recommendation,
    }


def save_members_csv(members: list[LineOAMember], path: str) -> None:
    """Write members list to CSV for inspection / manual import."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required")

    rows = [
        {
            "user_id":      m.user_id,
            "display_name": m.display_name,
            "language":     m.language,
            "tags":         "|".join(m.tags),
            "fetched_at":   m.fetched_at,
        }
        for m in members
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    _logger.info("save_members_csv: wrote %d rows to %s", len(rows), path)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _resolve_token(token: str | None) -> str:
    t = token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not t:
        raise ValueError(
            "LINE_CHANNEL_ACCESS_TOKEN ไม่ได้ตั้งค่า — "
            "ส่ง channel_access_token= หรือตั้ง env var"
        )
    return t


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }


def _get(url: str, params: dict | None = None, token: str = "") -> dict:
    return _request("GET", url, params=params, body=None, token=token)


def _post(url: str, body: dict, token: str = "") -> dict:
    return _request("POST", url, params=None, body=body, token=token)


def _request(
    method: str,
    url: str,
    params: dict | None,
    body: dict | None,
    token: str,
) -> dict:
    try:
        import requests as _req
    except ImportError:
        raise ImportError("'requests' package not installed")

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if method == "GET":
                resp = _req.get(url, headers=_headers(token),
                                params=params, timeout=_REQUEST_TIMEOUT)
            else:
                import json
                resp = _req.post(url, headers=_headers(token),
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 timeout=_REQUEST_TIMEOUT)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _logger.warning("LINE API: 429 rate-limit — retry %d/%d in %.1fs",
                                attempt, _MAX_RETRIES, delay)
                time.sleep(delay)
                continue

            if resp.status_code == 404:
                return {}  # user not found / blocked — not an error

            _logger.error("LINE API: HTTP %d %s %s — %s",
                          resp.status_code, method, url, resp.text[:200])
            resp.raise_for_status()

        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _logger.warning("LINE API: %s attempt %d/%d — retry in %.1fs",
                                type(exc).__name__, attempt, _MAX_RETRIES, delay)
                time.sleep(delay)

    raise RuntimeError(
        f"LINE API: all {_MAX_RETRIES} retries exhausted for {method} {url}. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Phone normalisation
# ---------------------------------------------------------------------------

def _normalise_phone(phone: str) -> str:
    """Normalise Thai phone to +66XXXXXXXXX format for matching."""
    if not phone:
        return ""
    p = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if p.startswith("0") and len(p) == 10:
        return "+66" + p[1:]
    if p.startswith("66") and len(p) == 11:
        return "+" + p
    return p

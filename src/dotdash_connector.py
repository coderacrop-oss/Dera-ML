"""dotdash_connector.py — DotDash (MeriPOS Loyalty) API client.

DotDash is MeriPOS's built-in CRM/loyalty module. Members sign up via LINE OA
and are identified by LINE userId.  This module:

1. Fetches all members via cursor-paginated API (AC-3.1)
2. Maps LINE userId → customer_id where available (AC-3.2)
3. Provides get_quality_report() comparing MeriPOS transactions vs DotDash members (AC-3.4)

Authentication
--------------
Set env var ``DOTDASH_API_TOKEN`` (session token from MeriPOS back-office).
Official partner API endpoint from CPOS TH not yet available; this client
supports both session-token (cookie) and Bearer-token auth patterns.

Usage
-----
    from dotdash_connector import DotDashClient
    client = DotDashClient()                    # reads token from env
    members_df = client.fetch_all_members()     # cursor-paginated, retries 3x
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL    = "https://api.mer-x.com/v2/loyalty"
_DEFAULT_PAGE_SIZE   = 100
_MAX_RETRIES         = 3
_RETRY_BASE_DELAY    = 1.0   # seconds; doubles each attempt
_REQUEST_TIMEOUT     = 15    # seconds


# ---------------------------------------------------------------------------
# Column mapping: DotDash API field → standard internal name
# ---------------------------------------------------------------------------
_API_COL_MAP: dict[str, str] = {
    # English API field names (from API response JSON)
    "id":                "dotdash_id",
    "lineUserId":        "line_user_id",
    "lineUserToken":     "line_user_id",   # alternative field name
    "line_user_id":      "line_user_id",
    "nickname":          "display_name",
    "phoneNumber":       "phone",
    "phone":             "phone",
    "visitCount":        "frequency",
    "totalSpend":        "monetary",
    "lastVisitDate":     "last_visit",
    "membershipLevel":   "tier",
    "segment":           "dotdash_segment",
    "points":            "points",
    # Thai field names (from Customer Report CSV, also used by API in some versions)
    "ชื่อเล่น":               "display_name",
    "เบอร์โทรศัพท์":          "phone",
    "จำนวนครั้งที่ใช้บริการ": "frequency",
    "ยอดใช้จ่าย":             "monetary",
    "ใช้บริการครั้งล่าสุด":    "last_visit",
    "ระดับสมาชิก":            "tier",
    "กลุ่มลูกค้า (EN)":       "dotdash_segment",
    "แต้มที่ใช้ได้":           "points",
}


# ---------------------------------------------------------------------------
# DotDashClient
# ---------------------------------------------------------------------------

@dataclass
class DotDashClient:
    """HTTP client for the DotDash loyalty API.

    Parameters
    ----------
    api_token : str, optional
        Session/bearer token. If None, reads ``DOTDASH_API_TOKEN`` from env.
    base_url : str
        API base URL. Defaults to ``api.mer-x.com/v2/loyalty``.
    page_size : int
        Members per page (cursor pagination). Default 100.
    timeout : int
        HTTP request timeout in seconds.
    """
    api_token: str | None = None
    base_url:  str        = _DEFAULT_BASE_URL
    page_size: int        = _DEFAULT_PAGE_SIZE
    timeout:   int        = _REQUEST_TIMEOUT

    def __post_init__(self) -> None:
        if self.api_token is None:
            self.api_token = os.environ.get("DOTDASH_API_TOKEN", "")
        if not self.api_token:
            _logger.warning("DOTDASH_API_TOKEN not set — API calls will be unauthenticated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all_members(self, store_id: str | None = None) -> pd.DataFrame:
        """Fetch all loyalty members via cursor pagination.

        Returns a normalised DataFrame with columns:
        customer_id, display_name, phone, frequency, monetary,
        recency, last_visit, tier, dotdash_segment, points

        ``customer_id`` is set to ``line_user_id`` when available,
        otherwise falls back to ``dotdash_id`` (AC-3.2).

        Parameters
        ----------
        store_id : str, optional
            Filter by branch/store. If None, fetches all stores.
        """
        raw_records: list[dict] = []
        cursor: str | None = None
        page  = 0

        while True:
            page += 1
            data = self._fetch_page(cursor=cursor, store_id=store_id)
            if data is None:
                _logger.error("fetch_all_members: page %d returned None — stopping", page)
                break

            records = data.get("data") or data.get("members") or data.get("items") or []
            raw_records.extend(records)

            next_cursor = data.get("nextCursor") or data.get("next_cursor") or data.get("cursor")
            has_more    = data.get("hasMore", data.get("has_more", bool(next_cursor)))

            _logger.debug("fetch_all_members: page %d — %d members, has_more=%s",
                          page, len(records), has_more)

            if not has_more or not next_cursor:
                break
            cursor = next_cursor

        _logger.info("fetch_all_members: fetched %d members total in %d pages",
                     len(raw_records), page)

        if not raw_records:
            return _empty_members_df()

        return _normalise_members(raw_records)

    def test_connection(self) -> bool:
        """Return True if the API responds with 200 on a minimal request."""
        try:
            data = self._fetch_page(cursor=None, store_id=None)
            return data is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, cursor: str | None, store_id: str | None) -> dict | None:
        """Fetch one page from /customers/list with retry logic.

        Retries up to _MAX_RETRIES times on rate-limit (429) or timeout.
        Uses exponential back-off: 1 s, 2 s, 4 s.

        Returns parsed JSON dict, or None on permanent failure.
        """
        try:
            import requests
        except ImportError:
            _logger.error("'requests' package not installed — cannot call DotDash API")
            return None

        url     = f"{self.base_url.rstrip('/')}/customers/list"
        headers = self._auth_headers()
        params: dict[str, Any] = {"limit": self.page_size}
        if cursor:
            params["cursor"] = cursor
        if store_id:
            params["storeId"] = store_id

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=headers, params=params,
                                    timeout=self.timeout)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    _logger.warning("DotDash API rate-limited (429) — retry %d/%d in %.1fs",
                                    attempt, _MAX_RETRIES, delay)
                    time.sleep(delay)
                    continue

                if resp.status_code == 401:
                    _logger.error("DotDash API: 401 Unauthorized — check DOTDASH_API_TOKEN")
                    return None

                _logger.warning("DotDash API: unexpected status %d on attempt %d",
                                resp.status_code, attempt)
                return None

            except Exception as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _logger.warning("DotDash API: %s on attempt %d/%d — retry in %.1fs",
                                type(exc).__name__, attempt, _MAX_RETRIES, delay)
                time.sleep(delay)

        _logger.error("DotDash API: all %d retries exhausted. Last error: %s",
                      _MAX_RETRIES, last_exc)
        return None

    def _auth_headers(self) -> dict[str, str]:
        """Build authentication headers.

        Tries Bearer token first; some MeriPOS versions use a session cookie.
        """
        if not self.api_token:
            return {}
        # MeriPOS back-office uses session tokens that may be passed as Bearer
        # or as a custom header depending on the API version.
        return {
            "Authorization": f"Bearer {self.api_token}",
            "X-Session-Token": self.api_token,
            "Content-Type": "application/json",
        }


# ---------------------------------------------------------------------------
# Normalisation helpers (AC-3.1 / AC-3.2)
# ---------------------------------------------------------------------------

def _normalise_members(raw: list[dict]) -> pd.DataFrame:
    """Convert raw API records to a normalised DataFrame.

    AC-3.2: ``customer_id`` is set to ``line_user_id`` when present,
    otherwise falls back to ``dotdash_id``. A warning is logged for
    any member without a LINE userId.
    """
    df = pd.DataFrame(raw)

    # Rename API fields → standard names
    rename = {k: v for k, v in _API_COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Deduplicate column names (some APIs send both English + Thai keys)
    df = df.loc[:, ~df.columns.duplicated()]

    # Numeric coercion
    for col in ("frequency", "monetary", "points"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Date parsing + recency (days since last_visit)
    if "last_visit" in df.columns:
        df["last_visit"] = pd.to_datetime(df["last_visit"],
                                          format="%d/%m/%Y",
                                          errors="coerce")
        # Try ISO format if the first parsing produced mostly NaT
        nat_rate = df["last_visit"].isna().mean()
        if nat_rate > 0.5:
            df["last_visit"] = pd.to_datetime(
                df["last_visit"].astype(str), errors="coerce"
            )
        today = pd.Timestamp.now().normalize()
        df["recency"] = (today - df["last_visit"]).dt.days.clip(lower=0)
    else:
        df["recency"] = None

    # Phone normalisation
    if "phone" in df.columns:
        df["phone"] = (
            df["phone"].astype(str)
            .str.replace(r"[-\s]", "", regex=True)
            .where(lambda s: ~s.isin({"-", "", "nan", "None", "NaN", "null"}),
                   other=None)
        )

    # AC-3.2: LINE userId → customer_id
    has_line = "line_user_id" in df.columns
    no_line_count = 0

    if has_line:
        # Use LINE userId where available; flag those that don't have one
        has_valid_line = df["line_user_id"].notna() & (df["line_user_id"] != "")
        no_line_count  = int((~has_valid_line).sum())

        df["customer_id"] = df["line_user_id"].where(
            has_valid_line,
            other=(
                "dotdash_" + df.get("dotdash_id", pd.Series(df.index, index=df.index)).astype(str)
            ),
        )
    else:
        no_line_count = len(df)
        id_col = "dotdash_id" if "dotdash_id" in df.columns else None
        if id_col:
            df["customer_id"] = "dotdash_" + df[id_col].astype(str)
        else:
            df["customer_id"] = "dotdash_" + df.index.astype(str)

    if no_line_count > 0:
        _logger.warning(
            "dotdash_connector: %d/%d members have no LINE userId — "
            "using internal DotDash ID as customer_id; LINE OA broadcast won't be possible",
            no_line_count, len(df),
        )

    # Ensure expected columns exist
    for col in ("display_name", "tier", "dotdash_segment", "points"):
        if col not in df.columns:
            df[col] = None

    return df[[
        "customer_id", "display_name", "phone",
        "frequency", "monetary", "recency", "last_visit",
        "tier", "dotdash_segment", "points",
    ] + (["line_user_id"] if has_line else [])].copy()


def _empty_members_df() -> pd.DataFrame:
    """Return an empty DataFrame with the standard member schema."""
    return pd.DataFrame(columns=[
        "customer_id", "display_name", "phone",
        "frequency", "monetary", "recency", "last_visit",
        "tier", "dotdash_segment", "points",
    ])


# ---------------------------------------------------------------------------
# AC-3.4 — Data Quality Report
# ---------------------------------------------------------------------------

def get_quality_report(meripos_df: pd.DataFrame, dotdash_df: pd.DataFrame) -> dict:
    """Compare MeriPOS transaction data with DotDash member data.

    Parameters
    ----------
    meripos_df : pd.DataFrame
        Raw MeriPOS Transaction Report (must contain 'Customer' or 'phone' column).
    dotdash_df : pd.DataFrame
        Normalised DotDash member DataFrame (from ingest_dotdash or fetch_all_members).

    Returns
    -------
    dict with keys:
        total_transactions, linked_transactions, linked_rate,
        total_members, members_with_spend, recommendation
    """
    total_tx = len(meripos_df)

    # Count transactions with a linked phone
    phone_col = "Customer" if "Customer" in meripos_df.columns else "phone"
    if phone_col in meripos_df.columns:
        null_sentinels = {"-", "", "nan", "None", "NaN", "null", "none"}
        linked_mask = (
            meripos_df[phone_col].notna()
            & ~meripos_df[phone_col].astype(str).str.strip().isin(null_sentinels)
        )
        linked_tx   = int(linked_mask.sum())
    else:
        linked_tx = 0

    linked_rate = round(linked_tx / total_tx, 4) if total_tx > 0 else 0.0

    total_members    = len(dotdash_df)
    members_with_spend = int((dotdash_df["monetary"] > 0).sum()) if "monetary" in dotdash_df.columns else 0

    recommendation = _pick_recommendation(linked_rate, total_members, members_with_spend, total_tx)

    return {
        "total_transactions":  total_tx,
        "linked_transactions": linked_tx,
        "linked_rate":         linked_rate,
        "total_members":       total_members,
        "members_with_spend":  members_with_spend,
        "recommendation":      recommendation,
    }


def _pick_recommendation(linked_rate: float, total_members: int,
                          members_with_spend: int, total_tx: int) -> str:
    """Generate a Thai-language recommendation based on data quality stats."""
    if linked_rate < 0.03:
        return (
            "สอนพนักงานสแกน QR สมาชิกทุกบิล — "
            "ปัจจุบันลิงค์ได้ต่ำกว่า 3% ทำให้ ML วิเคราะห์ลูกค้ารายบุคคลไม่ได้"
        )
    if linked_rate < 0.10:
        return (
            "ลองใช้ Prompt ที่จุด POS เพื่อเพิ่มอัตราสแกน QR สมาชิก — "
            f"ปัจจุบัน {linked_rate*100:.1f}% เป้าหมายควรถึง 30%"
        )
    if linked_rate < 0.30:
        return (
            f"อัตราลิงค์สมาชิก {linked_rate*100:.1f}% — ดีขึ้น แต่ยังมีช่องว่าง "
            "ลองจัดแคมเปญ Double Points เพื่อ incentivize การสแกนสมาชิก"
        )
    if members_with_spend < total_members * 0.5:
        return (
            f"ลิงค์ได้ {linked_rate*100:.1f}% แต่สมาชิก {members_with_spend}/{total_members} คนมียอดซื้อ — "
            "แนะนำ Win-Back campaign สำหรับสมาชิกที่ไม่มีธุรกรรม"
        )
    return (
        f"ข้อมูลดี: ลิงค์ได้ {linked_rate*100:.1f}% สมาชิก {total_members} คน — "
        "พร้อมรัน ML campaign แบบ full mode ได้เลย"
    )

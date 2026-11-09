"""meta_ads_client.py — Meta Ads API client for campaign execution.

Implements the 6-step pipeline for running a Meta (Facebook/Instagram) ad campaign:

    1. upload_custom_audience(phone_list)    → audience_id
    2. create_lookalike(audience_id, ...)    → lookalike_id
    3. create_campaign(name, objective, budget_thb) → campaign_id
    4. create_ad_set(campaign_id, audience_id, budget) → ad_set_id
    5. create_creative(image_url, headline, body, cta) → creative_id
    6. create_ad(ad_set_id, creative_id)    → ad_id

Plus:
    get_ad_insights(ad_id) → {impressions, clicks, ctr, conversions, spend_thb}

Authentication
--------------
Set env vars:
    META_ACCESS_TOKEN   — permanent page/system-user token
    META_AD_ACCOUNT_ID  — ad account ID (without "act_" prefix)
    META_PAGE_ID        — Facebook Page ID for ad creative

Usage
-----
    from meta_ads_client import MetaAdsClient

    client = MetaAdsClient()
    audience_id  = client.upload_custom_audience(phone_list=["0812345678", ...])
    lookalike_id = client.create_lookalike(audience_id)
    campaign_id  = client.create_campaign("Win-Back Nov 2026", "CONVERSIONS", 5000)
    ad_set_id    = client.create_ad_set(campaign_id, audience_id, 5000)
    creative_id  = client.create_creative("https://...", "โปรพิเศษ", "ลดทันที 20%", "SHOP_NOW")
    ad_id        = client.create_ad(ad_set_id, creative_id)
    insights     = client.get_ad_insights(ad_id)

AC-5.2 requirements met.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)

_META_GRAPH_URL = "https://graph.facebook.com/v19.0"
_MAX_RETRIES    = 3
_RETRY_DELAY    = 2.0
_REQUEST_TIMEOUT = 30

# THB to USD conversion for Meta (Meta API takes USD)
_THB_PER_USD = 36.0


# ---------------------------------------------------------------------------
# MetaAdsClient
# ---------------------------------------------------------------------------

@dataclass
class MetaAdsClient:
    """Client for the Meta (Facebook/Instagram) Ads API.

    Parameters
    ----------
    access_token : str, optional
        Meta permanent access token. Falls back to env ``META_ACCESS_TOKEN``.
    ad_account_id : str, optional
        Ad account ID (without "act_" prefix). Falls back to ``META_AD_ACCOUNT_ID``.
    page_id : str, optional
        Facebook Page ID. Falls back to ``META_PAGE_ID``.
    """
    access_token:  str | None = None
    ad_account_id: str | None = None
    page_id:       str | None = None

    def __post_init__(self) -> None:
        self.access_token  = self.access_token  or os.environ.get("META_ACCESS_TOKEN",  "")
        self.ad_account_id = self.ad_account_id or os.environ.get("META_AD_ACCOUNT_ID", "")
        self.page_id       = self.page_id       or os.environ.get("META_PAGE_ID",       "")

        if not self.access_token:
            _logger.warning("META_ACCESS_TOKEN not set — API calls will fail")
        if not self.ad_account_id:
            _logger.warning("META_AD_ACCOUNT_ID not set — API calls will fail")

    # ------------------------------------------------------------------
    # Step 1: Upload Custom Audience (phone list → hashed)
    # ------------------------------------------------------------------

    def upload_custom_audience(
        self,
        phone_list: list[str],
        audience_name: str = "CRM Custom Audience",
    ) -> str:
        """Create a Meta Custom Audience from a phone number list.

        Phones are SHA-256 hashed before upload (Meta requirement).
        Returns the audience_id string.
        """
        # 1a. Create the audience container
        payload = {
            "name":        audience_name,
            "subtype":     "CUSTOM",
            "description": "Uploaded from ML campaign engine",
            "customer_file_source": "USER_PROVIDED_ONLY",
        }
        resp = self._post(f"act_{self.ad_account_id}/customaudiences", data=payload)
        audience_id = resp["id"]
        _logger.info("Meta: created custom audience %s", audience_id)

        # 1b. Upload hashed phones in one batch (Meta supports up to 10 k)
        hashed = [_hash_phone(p) for p in phone_list if p]
        schema_payload = {
            "schema": ["PHONE"],
            "data":   [[h] for h in hashed],
        }
        self._post(
            f"{audience_id}/users",
            data={"payload": str(schema_payload).replace("'", '"')},
        )
        _logger.info("Meta: uploaded %d hashed phones to audience %s", len(hashed), audience_id)
        return audience_id

    # ------------------------------------------------------------------
    # Step 2: Create Lookalike Audience
    # ------------------------------------------------------------------

    def create_lookalike(
        self,
        audience_id: str,
        country: str = "TH",
        ratio: float = 0.01,
    ) -> str:
        """Create a 1% Lookalike Audience from a custom audience.  Returns lookalike_id."""
        payload = {
            "name":            f"Lookalike ({country}, {ratio*100:.0f}%)",
            "origin_audience_id": audience_id,
            "lookalike_spec": f'{{"country":"{country}","ratio":{ratio},"type":"similarity"}}',
        }
        resp = self._post(f"act_{self.ad_account_id}/customaudiences", data=payload)
        lookalike_id = resp["id"]
        _logger.info("Meta: created lookalike audience %s from %s", lookalike_id, audience_id)
        return lookalike_id

    # ------------------------------------------------------------------
    # Step 3: Create Campaign
    # ------------------------------------------------------------------

    def create_campaign(
        self,
        name: str,
        objective: str = "CONVERSIONS",
        budget_thb: float = 1000.0,
    ) -> str:
        """Create a Meta campaign.  Returns campaign_id.

        ``budget_thb`` is a hint stored in the name — actual budget set in ad set.
        """
        payload = {
            "name":             name,
            "objective":        objective,
            "status":           "PAUSED",   # start paused; activate after review
            "special_ad_categories": "[]",
        }
        resp = self._post(f"act_{self.ad_account_id}/campaigns", data=payload)
        campaign_id = resp["id"]
        _logger.info("Meta: created campaign %s (id=%s, budget_hint=฿%.0f)",
                     name, campaign_id, budget_thb)
        return campaign_id

    # ------------------------------------------------------------------
    # Step 4: Create Ad Set
    # ------------------------------------------------------------------

    def create_ad_set(
        self,
        campaign_id: str,
        audience_id: str,
        budget_thb: float,
        start_time: str | None = None,
    ) -> str:
        """Create an ad set targeting the given audience.  Returns ad_set_id.

        ``budget_thb`` is converted to USD cents for Meta API.
        """
        budget_usd_cents = int(budget_thb / _THB_PER_USD * 100)

        payload = {
            "name":             f"Ad Set — ฿{budget_thb:.0f}",
            "campaign_id":      campaign_id,
            "daily_budget":     str(budget_usd_cents),
            "billing_event":    "IMPRESSIONS",
            "optimization_goal": "REACH",
            "targeting":        f'{{"custom_audiences":[{{"id":"{audience_id}"}}],"geo_locations":{{"countries":["TH"]}}}}',
            "status":           "PAUSED",
        }
        if start_time:
            payload["start_time"] = start_time

        resp = self._post(f"act_{self.ad_account_id}/adsets", data=payload)
        ad_set_id = resp["id"]
        _logger.info("Meta: created ad set %s (campaign=%s, budget=฿%.0f)",
                     ad_set_id, campaign_id, budget_thb)
        return ad_set_id

    # ------------------------------------------------------------------
    # Step 5: Create Ad Creative
    # ------------------------------------------------------------------

    def create_creative(
        self,
        image_url: str,
        headline: str,
        body: str,
        cta: str = "SHOP_NOW",
    ) -> str:
        """Create an ad creative.  Returns creative_id."""
        link_data = (
            f'{{"image_url":"{image_url}",'
            f'"message":"{body}",'
            f'"link":"https://www.facebook.com/{self.page_id}",'
            f'"name":"{headline}",'
            f'"call_to_action":{{"type":"{cta}"}}}}'
        )
        payload = {
            "name":        f"Creative — {headline[:40]}",
            "object_story_spec": (
                f'{{"page_id":"{self.page_id}",'
                f'"link_data":{link_data}}}'
            ),
        }
        resp = self._post(f"act_{self.ad_account_id}/adcreatives", data=payload)
        creative_id = resp["id"]
        _logger.info("Meta: created creative %s ('%s')", creative_id, headline[:40])
        return creative_id

    # ------------------------------------------------------------------
    # Step 6: Create Ad
    # ------------------------------------------------------------------

    def create_ad(self, ad_set_id: str, creative_id: str, name: str = "Ad") -> str:
        """Create an ad linking an ad set to a creative.  Returns ad_id."""
        payload = {
            "name":        name,
            "adset_id":    ad_set_id,
            "creative":    f'{{"creative_id":"{creative_id}"}}',
            "status":      "PAUSED",
        }
        resp = self._post(f"act_{self.ad_account_id}/ads", data=payload)
        ad_id = resp["id"]
        _logger.info("Meta: created ad %s (adset=%s, creative=%s)",
                     ad_id, ad_set_id, creative_id)
        return ad_id

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def get_ad_insights(
        self,
        ad_id: str,
        date_preset: str = "last_30d",
    ) -> dict:
        """Return performance metrics for an ad.

        Returns
        -------
        dict:
            ``{impressions, clicks, ctr, conversions, spend_thb}``
        """
        params = {
            "fields": "impressions,clicks,ctr,conversions,spend",
            "date_preset": date_preset,
        }
        data = self._get(f"{ad_id}/insights", params=params)
        raw = data.get("data", [{}])[0] if data.get("data") else {}

        impressions = int(raw.get("impressions", 0) or 0)
        clicks      = int(raw.get("clicks", 0) or 0)
        ctr         = float(raw.get("ctr", 0) or 0)
        conversions = int(
            sum(int(c.get("value", 0)) for c in raw.get("conversions", []))
            if isinstance(raw.get("conversions"), list)
            else int(raw.get("conversions", 0) or 0)
        )
        spend_usd   = float(raw.get("spend", 0) or 0)
        spend_thb   = round(spend_usd * _THB_PER_USD, 2)

        return {
            "impressions": impressions,
            "clicks":      clicks,
            "ctr":         ctr,
            "conversions": conversions,
            "spend_thb":   spend_thb,
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, data: dict) -> dict:
        """POST to Meta Graph API with retry.  Returns parsed JSON."""
        return self._request("POST", endpoint, data=data)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """GET from Meta Graph API with retry.  Returns parsed JSON."""
        return self._request("GET", endpoint, params=params)

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        try:
            import requests as _requests
        except ImportError:
            _logger.error("'requests' package not installed — cannot call Meta API")
            return {}

        url = f"{_META_GRAPH_URL}/{endpoint}"
        base_params = {"access_token": self.access_token}
        if params:
            base_params.update(params)

        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if method == "POST":
                    resp = _requests.post(url, params=base_params, data=data,
                                          timeout=_REQUEST_TIMEOUT)
                else:
                    resp = _requests.get(url, params=base_params, timeout=_REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    delay = _RETRY_DELAY * (2 ** (attempt - 1))
                    _logger.warning("Meta API: 429 rate-limit — retry %d/%d in %.1fs",
                                    attempt, _MAX_RETRIES, delay)
                    time.sleep(delay)
                    continue

                _logger.error("Meta API: HTTP %d on %s %s — %s",
                              resp.status_code, method, endpoint, resp.text[:200])
                resp.raise_for_status()

            except Exception as exc:
                last_exc = exc
                delay = _RETRY_DELAY * (2 ** (attempt - 1))
                _logger.warning("Meta API: %s on attempt %d/%d — retry in %.1fs",
                                type(exc).__name__, attempt, _MAX_RETRIES, delay)
                if attempt < _MAX_RETRIES:
                    time.sleep(delay)

        raise RuntimeError(
            f"Meta API: all {_MAX_RETRIES} retries exhausted for {method} {endpoint}. "
            f"Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hash_phone(phone: str) -> str:
    """Normalise and SHA-256 hash a phone number per Meta's spec."""
    # Strip spaces, dashes; ensure Thai numbers start with country code
    normalized = phone.strip().replace(" ", "").replace("-", "")
    if normalized.startswith("0") and len(normalized) == 10:
        normalized = "+66" + normalized[1:]  # Thai mobile: 0812... → +66812...
    return hashlib.sha256(normalized.encode()).hexdigest()

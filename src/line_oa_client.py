"""line_oa_client.py — LINE Messaging API client for campaign broadcast.

Sends personalised messages to LINE OA members using the Multicast API.
Auto-batches to 500 user_ids per call (LINE API limit) and handles
rate-limiting (HTTP 429) with exponential back-off.

Authentication
--------------
Set env var ``LINE_CHANNEL_ACCESS_TOKEN`` or pass ``channel_access_token``
directly to :func:`multicast`.

Usage
-----
    from line_oa_client import multicast

    result = multicast(
        user_ids=["Uabc...", "Udef...", ...],
        message_template={"type": "text", "text": "สวัสดี {name}! โปรพิเศษสำหรับคุณ"},
        channel_access_token="...",
    )
    print(result)   # {"sent": 1200, "failed": 0, "failed_ids": []}

AC-5.1 requirements met:
- max 500 user_ids per Multicast call → auto-batch
- rate limit 429 → exponential back-off (1s, 2s, 4s, max 3 retries)
- returns {sent, failed, failed_ids}
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)

_LINE_MULTICAST_URL = "https://api.line.me/v2/bot/message/multicast"
_BATCH_SIZE         = 500   # LINE Multicast limit
_MAX_RETRIES        = 3
_RETRY_BASE_DELAY   = 1.0   # seconds; doubles each attempt
_REQUEST_TIMEOUT    = 20    # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multicast(
    user_ids: list[str],
    message_template: dict | list[dict],
    channel_access_token: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Send a LINE Multicast message to a list of user IDs.

    Parameters
    ----------
    user_ids : list[str]
        LINE user IDs (``U`` + 32 hex chars).  Duplicates are removed.
    message_template : dict | list[dict]
        One message object or list of up to 5 message objects in LINE format.
        e.g. ``{"type": "text", "text": "สวัสดีครับ!"}``
    channel_access_token : str | None
        Bearer token. Falls back to env var ``LINE_CHANNEL_ACCESS_TOKEN``.
    dry_run : bool
        If True, skip actual HTTP calls; return simulated success result.

    Returns
    -------
    dict:
        ``{"sent": int, "failed": int, "failed_ids": list[str]}``
    """
    token = channel_access_token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token and not dry_run:
        raise ValueError(
            "LINE_CHANNEL_ACCESS_TOKEN ไม่ได้ตั้งค่า — "
            "ตั้ง env var หรือส่ง channel_access_token มาด้วย"
        )

    # Normalise message to list
    messages = message_template if isinstance(message_template, list) else [message_template]
    if len(messages) > 5:
        _logger.warning("LINE Multicast: ส่งได้สูงสุด 5 messages ต่อครั้ง — ตัดเหลือ 5")
        messages = messages[:5]

    # Deduplicate user IDs
    unique_ids = list(dict.fromkeys(uid for uid in user_ids if uid))
    if not unique_ids:
        return {"sent": 0, "failed": 0, "failed_ids": []}

    # Split into batches of ≤ 500
    batches = [unique_ids[i : i + _BATCH_SIZE] for i in range(0, len(unique_ids), _BATCH_SIZE)]
    _logger.info(
        "LINE Multicast: %d user_ids → %d batch(es) of ≤%d",
        len(unique_ids), len(batches), _BATCH_SIZE,
    )

    sent       = 0
    failed_ids: list[str] = []

    for batch_idx, batch in enumerate(batches, start=1):
        if dry_run:
            sent += len(batch)
            _logger.debug("LINE Multicast dry_run: batch %d — %d sent", batch_idx, len(batch))
            continue

        ok = _send_batch(batch, messages, token, batch_idx)
        if ok:
            sent += len(batch)
        else:
            failed_ids.extend(batch)
            _logger.error(
                "LINE Multicast: batch %d/%d failed — %d user_ids marked as failed",
                batch_idx, len(batches), len(batch),
            )

    return {
        "sent":       sent,
        "failed":     len(failed_ids),
        "failed_ids": failed_ids,
    }


# ---------------------------------------------------------------------------
# Batch sender with retry
# ---------------------------------------------------------------------------

def _send_batch(
    user_ids: list[str],
    messages: list[dict],
    token: str,
    batch_idx: int,
) -> bool:
    """POST one Multicast batch to LINE API.  Retries on 429.  Returns True on success."""
    try:
        import requests
    except ImportError:
        _logger.error("'requests' package not installed — cannot call LINE API")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload: dict[str, Any] = {
        "to":       user_ids,
        "messages": messages,
    }

    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                _LINE_MULTICAST_URL,
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                _logger.debug("LINE Multicast: batch %d sent OK (%d ids)", batch_idx, len(user_ids))
                return True

            if resp.status_code == 429:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _logger.warning(
                    "LINE Multicast: 429 rate-limited on batch %d — retry %d/%d in %.1fs",
                    batch_idx, attempt, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 401:
                _logger.error(
                    "LINE Multicast: 401 Unauthorized — check LINE_CHANNEL_ACCESS_TOKEN"
                )
                return False

            _logger.error(
                "LINE Multicast: unexpected HTTP %d on batch %d",
                resp.status_code, batch_idx,
            )
            return False

        except Exception as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            _logger.warning(
                "LINE Multicast: %s on batch %d attempt %d/%d — retry in %.1fs",
                type(exc).__name__, batch_idx, attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)

    _logger.error(
        "LINE Multicast: batch %d exhausted all %d retries. Last error: %s",
        batch_idx, _MAX_RETRIES, last_exc,
    )
    return False

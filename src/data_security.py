"""data_security.py — Privacy and security primitives for the ML campaign pipeline.

Implements:
  1. customer_id hashing (PDPA data minimization)
  2. Column allowlist filtering (reject PII columns at ingestion)
  3. APIKeyManager — per-partner scoped key management (file-based)
  4. AuditLogger — append-only audit trail for all significant events
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component 1: customer_id hashing
# ---------------------------------------------------------------------------

def hash_customer_id(raw_id: str, store_salt: str) -> str:
    """One-way hash of customer_id using per-store salt.

    Uses HMAC-SHA256 (more secure than plain SHA256 for this use case — the
    salt acts as the HMAC key, preventing length-extension attacks and making
    it infeasible to brute-force without knowledge of the salt).

    Parameters
    ----------
    raw_id : str
        The raw customer identifier (e.g. phone number, email, or opaque id).
    store_salt : str
        A per-store secret. Must never be hardcoded — should come from an
        environment variable so different stores have different salts.

    Returns
    -------
    str
        64-character lowercase hex string. Deterministic: same raw_id + same
        salt always returns the same hash.
    """
    if not store_salt:
        raise ValueError(
            "store_salt must not be empty. Pass a per-store secret or set the "
            "STORE_SALT environment variable."
        )
    mac = hmac.new(store_salt.encode("utf-8"), raw_id.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


def hash_dataframe_ids(
    df: pd.DataFrame,
    id_col: str = "customer_id",
    store_salt: str = "",
) -> pd.DataFrame:
    """Apply hash_customer_id to every row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Must contain *id_col*.
    id_col : str, default "customer_id"
        Name of the column containing raw customer identifiers.
    store_salt : str, default ""
        Per-store secret. If empty string, the value is read from the
        ``STORE_SALT`` environment variable. If that is also unset a
        ``ValueError`` is raised.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *id_col* replaced by hashed values.

    Raises
    ------
    ValueError
        If neither *store_salt* nor ``STORE_SALT`` env var is set.
    KeyError
        If *id_col* is not present in *df*.
    """
    if not store_salt:
        store_salt = os.environ.get("STORE_SALT", "")
    if not store_salt:
        raise ValueError(
            "No store_salt provided and STORE_SALT environment variable is not set. "
            "Set STORE_SALT to a per-store secret before hashing customer IDs."
        )
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame.")

    df = df.copy()
    n_total = len(df)
    n_already_hashed = int(df[id_col].astype(str).apply(is_already_hashed).sum())
    n_to_hash = n_total - n_already_hashed

    if n_already_hashed > 0:
        logger.info(
            "hash_dataframe_ids: %d/%d ids appear already hashed — skipping those rows.",
            n_already_hashed,
            n_total,
        )

    def _hash_if_needed(val: object) -> str:
        s = str(val)
        if is_already_hashed(s):
            return s
        return hash_customer_id(s, store_salt)

    df[id_col] = df[id_col].astype(str).apply(_hash_if_needed)
    logger.info("hash_dataframe_ids: hashed %d customer ids.", n_to_hash)
    return df


def is_already_hashed(customer_id: str) -> bool:
    """Heuristic check: returns True if *customer_id* looks like a 64-char hex string.

    Prevents double-hashing when a partner has already sent pre-hashed ids.
    A 64-character lowercase hex string is the expected output of SHA-256 /
    HMAC-SHA256.
    """
    return bool(re.fullmatch(r"[0-9a-f]{64}", customer_id))


# ---------------------------------------------------------------------------
# Component 2: Column allowlist filtering
# ---------------------------------------------------------------------------

# These are the ONLY columns the pipeline is permitted to process.
# Any column not in this set is treated as potentially PII and dropped.
ALLOWED_COLUMNS: frozenset[str] = frozenset({
    "customer_id", "timestamp", "transaction_date", "created_at",
    "total_price", "amount", "unit_price", "quantity",
    "product_name", "category",
    "transaction_id", "invoice_id", "receipt_id",
    "is_treated", "post_promo_visit",
    "promo_applied", "promo_type",   # used by promo_advisor
})


def filter_columns(
    df: pd.DataFrame,
    warn: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop all columns not in ALLOWED_COLUMNS.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe, potentially containing PII columns.
    warn : bool, default True
        If True, emit a logger.warning for each dropped column name.
        Column *values* are never logged.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(filtered_df, list_of_dropped_column_names)``
        The returned dataframe only contains columns that are in
        ALLOWED_COLUMNS *and* present in the input.
    """
    present = set(df.columns)
    keep = [c for c in df.columns if c in ALLOWED_COLUMNS]
    dropped = sorted(present - set(keep))

    if dropped and warn:
        logger.warning(
            "filter_columns: dropped %d non-allowlisted column(s): %s",
            len(dropped),
            dropped,
        )

    return df[keep].copy(), dropped


# ---------------------------------------------------------------------------
# Component 3: APIKeyManager
# ---------------------------------------------------------------------------

_KEY_SIGNING_SECRET_ENV = "KEY_SIGNING_SECRET"
_DEFAULT_REGISTRY_PATH = "data/security/api_keys.json"


def _hmac_key(signing_secret: str, plaintext_key: str) -> str:
    """Compute HMAC-SHA256 of *plaintext_key* using *signing_secret*."""
    return hmac.new(
        signing_secret.encode("utf-8"),
        plaintext_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


class APIKeyManager:
    """Per-partner scoped API key management backed by a JSON file.

    Key schema stored in the registry::

        {
          "key_id": "pk_live_<random_8_chars>",
          "key_hash": "<hmac_sha256_of_key>",   # only hash stored, never plaintext
          "partner_id": "partner_A",
          "store_scope": ["store_01"],           # stores this key may access
          "created_at": "2026-11-05T10:30:00Z",
          "revoked_at": null                     # ISO timestamp or null
        }

    Usage::

        km = APIKeyManager()
        key, meta = km.generate_key("partner_A", store_scope=["store_01"])
        # key is shown ONCE — partner must save it, we only keep the hash

        meta = km.validate_key(key, required_store="store_01")
        # returns key meta if valid+not-revoked+store-in-scope, else raises

        km.revoke_key("pk_live_xxxx")

    Thread safety
    -------------
    All mutating methods (generate_key, revoke_key) use an in-process lock
    plus an atomic write (write to tmp, then os.replace) to avoid partial
    writes. This is safe for single-process deployments; a database-backed
    implementation is recommended for multi-process / distributed deployments.
    """

    def __init__(
        self,
        registry_path: str = _DEFAULT_REGISTRY_PATH,
        test_mode: bool = False,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.test_mode = test_mode
        self._lock = threading.Lock()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_key(
        self,
        partner_id: str,
        store_scope: list[str],
    ) -> tuple[str, dict]:
        """Generate a new scoped API key for *partner_id*.

        Parameters
        ----------
        partner_id : str
            Identifier for the partner (e.g. ``"partner_A"``).
        store_scope : list[str]
            List of store IDs this key is permitted to access.

        Returns
        -------
        tuple[str, dict]
            ``(plaintext_key, metadata)`` — the plaintext key is returned
            exactly once and is NOT stored. The caller must relay it to the
            partner securely.

        Raises
        ------
        RuntimeError
            If ``KEY_SIGNING_SECRET`` environment variable is not set.
        """
        signing_secret = self._get_signing_secret()
        prefix = "pk_test_" if self.test_mode else "pk_live_"
        random_part = secrets.token_urlsafe(24)
        plaintext_key = f"{prefix}{random_part}"
        # Derive a stable key_id from the first 8 chars of the random part
        key_id = f"{prefix}{random_part[:8]}"
        key_hash = _hmac_key(signing_secret, plaintext_key)

        meta: dict = {
            "key_id": key_id,
            "key_hash": key_hash,
            "partner_id": partner_id,
            "store_scope": list(store_scope),
            "created_at": _utcnow_iso(),
            "revoked_at": None,
        }

        with self._lock:
            registry = self._read_registry()
            registry.append(meta)
            self._write_registry(registry)

        logger.info(
            "APIKeyManager: generated key_id=%s for partner=%s scope=%s",
            key_id,
            partner_id,
            store_scope,
        )
        return plaintext_key, {k: v for k, v in meta.items() if k != "key_hash"}

    def validate_key(
        self,
        plaintext_key: str,
        required_store: Optional[str] = None,
    ) -> dict:
        """Validate a plaintext API key.

        Parameters
        ----------
        plaintext_key : str
            The key as supplied by the partner.
        required_store : str or None
            If provided, the key must have this store in its ``store_scope``.

        Returns
        -------
        dict
            Key metadata (without ``key_hash``) if the key is valid, active,
            and (if *required_store* is set) in scope.

        Raises
        ------
        ValueError
            If the key is not found, has been revoked, or is out of scope.
        RuntimeError
            If ``KEY_SIGNING_SECRET`` environment variable is not set.
        """
        signing_secret = self._get_signing_secret()
        candidate_hash = _hmac_key(signing_secret, plaintext_key)

        registry = self._read_registry()
        matched: Optional[dict] = None
        for entry in registry:
            if hmac.compare_digest(entry["key_hash"], candidate_hash):
                matched = entry
                break

        if matched is None:
            raise ValueError("API key not recognised.")

        if matched.get("revoked_at") is not None:
            raise ValueError(
                f"API key has been revoked at {matched['revoked_at']}."
            )

        if required_store is not None:
            scope = matched.get("store_scope", [])
            if required_store not in scope:
                raise ValueError(
                    f"API key does not grant access to store '{required_store}'. "
                    f"Authorised stores: {scope}."
                )

        # Return metadata without the hash
        return {k: v for k, v in matched.items() if k != "key_hash"}

    def revoke_key(self, key_id: str) -> None:
        """Mark a key as revoked.

        Sets ``revoked_at`` to the current UTC timestamp. The record is
        preserved (not deleted) to maintain the audit trail.

        Parameters
        ----------
        key_id : str
            The ``key_id`` of the key to revoke (e.g. ``"pk_live_xxxx"``).

        Raises
        ------
        KeyError
            If no key with *key_id* is found in the registry.
        """
        with self._lock:
            registry = self._read_registry()
            for entry in registry:
                if entry["key_id"] == key_id:
                    if entry["revoked_at"] is not None:
                        logger.warning(
                            "APIKeyManager: key_id=%s is already revoked.", key_id
                        )
                        return
                    entry["revoked_at"] = _utcnow_iso()
                    self._write_registry(registry)
                    logger.info("APIKeyManager: revoked key_id=%s", key_id)
                    return
            raise KeyError(f"No key found with key_id='{key_id}'.")

    def list_keys(self, partner_id: Optional[str] = None) -> list[dict]:
        """List all registered keys, optionally filtered by *partner_id*.

        Returns
        -------
        list[dict]
            Key metadata records. ``key_hash`` is excluded from every record
            to prevent hash-based brute-force lookups.
        """
        registry = self._read_registry()
        if partner_id is not None:
            registry = [e for e in registry if e.get("partner_id") == partner_id]
        return [{k: v for k, v in entry.items() if k != "key_hash"} for entry in registry]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_registry(self) -> list[dict]:
        """Read and return the registry from disk. Returns [] if file is missing."""
        if not self.registry_path.exists():
            return []
        with open(self.registry_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_registry(self, registry: list[dict]) -> None:
        """Atomically write the registry to disk using a temp file + os.replace."""
        parent = self.registry_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(registry, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.replace(tmp_path, self.registry_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _get_signing_secret() -> str:
        """Return KEY_SIGNING_SECRET from env or raise RuntimeError."""
        secret = os.environ.get(_KEY_SIGNING_SECRET_ENV, "")
        if not secret:
            raise RuntimeError(
                f"Environment variable '{_KEY_SIGNING_SECRET_ENV}' is not set. "
                "Set it to a strong random secret before using APIKeyManager."
            )
        return secret


# ---------------------------------------------------------------------------
# Component 4: AuditLogger
# ---------------------------------------------------------------------------

EVENT_TYPES = frozenset({
    "data_upload",      # partner uploads CSV
    "inference_run",    # pipeline runs for a store
    "api_call",         # any API endpoint called
    "key_generated",    # new API key created
    "key_revoked",      # key revoked
    "data_deleted",     # files deleted (retention or erasure)
    "erasure_request",  # right-to-erasure request received
    "column_dropped",   # PII columns dropped at ingestion (logs column names only)
})

_DEFAULT_AUDIT_LOG_PATH = "data/security/audit.log"


class AuditLogger:
    """Append-only audit log for PDPA compliance.

    Writes one JSON object per line (NDJSON) to the configured log file.

    Event schema::

        {
          "ts": "2026-11-05T10:30:00Z",    # UTC ISO timestamp
          "event": "data_upload",           # one of EVENT_TYPES
          "partner_id": "partner_A",
          "store_id": "store_01",
          "actor": "api_key:pk_live_xxxx", # or "system" or "user:<name>"
          "detail": {...}                   # event-specific metadata (NO PII values)
        }

    Thread safety
    -------------
    Uses a per-instance threading.Lock so concurrent writes from multiple
    threads in the same process are serialised. For multi-process deployments
    the OS-level atomic write (append mode on POSIX) provides sufficient
    safety for most scenarios, but a database-backed log is recommended at
    high throughput.
    """

    def __init__(self, log_path: str = _DEFAULT_AUDIT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        event: str,
        partner_id: str = "",
        store_id: str = "",
        actor: str = "system",
        detail: Optional[dict] = None,
    ) -> None:
        """Append one audit event to the log file.

        Parameters
        ----------
        event : str
            Event type. Should be one of ``EVENT_TYPES`` but unknown values
            are accepted (logged with a warning) to avoid swallowing errors.
        partner_id : str, optional
            Partner identifier associated with this event.
        store_id : str, optional
            Store identifier associated with this event.
        actor : str, default "system"
            Who triggered the event: ``"system"``, ``"api_key:pk_live_xxxx"``,
            or ``"user:<name>"``. Never put actual data values here.
        detail : dict or None, optional
            Event-specific metadata. Must NOT contain PII values — only
            counts, identifiers, column names, and timestamps.

        Notes
        -----
        This method never raises; errors are emitted to the Python logger
        so a logging failure does not break the pipeline.
        """
        if event not in EVENT_TYPES:
            logger.warning("AuditLogger.log: unknown event type '%s'.", event)

        record = {
            "ts": _utcnow_iso(),
            "event": event,
            "partner_id": partner_id,
            "store_id": store_id,
            "actor": actor,
            "detail": detail or {},
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        try:
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError as exc:
            logger.error("AuditLogger: failed to write audit event: %s", exc)

    def tail(self, n: int = 50) -> list[dict]:
        """Return the last *n* audit events.

        Reads the entire log file and returns the final *n* lines parsed as
        dicts. Suitable for debugging/display; for production monitoring use
        a dedicated log aggregation tool.

        Parameters
        ----------
        n : int, default 50
            Number of most-recent events to return.

        Returns
        -------
        list[dict]
            Parsed event records (empty list if log file does not exist).
        """
        if not self.log_path.exists():
            return []

        events: list[dict] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    events.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    logger.warning("AuditLogger.tail: skipping malformed line.")

        return events[-n:]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Return current UTC time as an ISO 8601 string with 'Z' suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

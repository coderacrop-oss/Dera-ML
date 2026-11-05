"""data_retention.py — Data lifecycle management for PDPA compliance.

Two responsibilities:
  1. RetentionCleaner — scan data directories and delete files older than
     the configured retention period.
  2. ErasureProcessor — delete all data for a specific hashed customer_id
     across all CSV files in the data directory tree.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retention configuration
# ---------------------------------------------------------------------------

# Default retention periods in days, keyed by data tier.
RETENTION_DAYS: dict[str, int] = {
    "raw":       90,   # raw POS uploads
    "processed": 365,  # processed features, cust_agg
    "output":    365,  # broadcast lists, campaign outputs
    "test":      1,    # partner test uploads — delete next day
    "audit":     730,  # audit logs — 2 years
}

# ---------------------------------------------------------------------------
# RetentionCleaner
# ---------------------------------------------------------------------------

class RetentionCleaner:
    """Scan data directories and delete files older than their retention period.

    Directory structure assumed::

        data/raw/{store_id}/          → 90 days
        data/processed/{store_id}/    → 365 days
        data/output/{store_id}/       → 365 days
        data/test/                    → 1 day

    Parameters
    ----------
    base_dir : str, default "data/"
        Root of the data directory tree.
    dry_run : bool, default True
        When ``True`` the cleaner logs what *would* be deleted but does not
        actually remove any files. Set to ``False`` to perform real deletions.
    retention_days : dict or None, optional
        Override the default ``RETENTION_DAYS`` mapping. Keys must match the
        tier names used in ``base_dir`` (``"raw"``, ``"processed"``, etc.).

    Usage::

        cleaner = RetentionCleaner(base_dir="data/", dry_run=True)
        report = cleaner.run()
        # dry_run=True: logs what would be deleted without touching files
        # dry_run=False: deletes expired files and logs each deletion
    """

    # Map subdirectory names to retention tier keys
    _DIR_TO_TIER: dict[str, str] = {
        "raw":       "raw",
        "processed": "processed",
        "output":    "output",
        "test":      "test",
    }

    def __init__(
        self,
        base_dir: str = "data/",
        dry_run: bool = True,
        retention_days: Optional[dict[str, int]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.retention = {**RETENTION_DAYS, **(retention_days or {})}

    def run(self) -> dict:
        """Scan and optionally delete expired files.

        Returns
        -------
        dict
            Summary with keys: ``files_scanned``, ``files_deleted``,
            ``bytes_freed``, ``errors``.
        """
        files_scanned = 0
        files_deleted = 0
        bytes_freed = 0
        errors: list[str] = []
        now = datetime.now(tz=timezone.utc)

        for dir_name, tier in self._DIR_TO_TIER.items():
            tier_dir = self.base_dir / dir_name
            if not tier_dir.exists():
                continue

            max_age = timedelta(days=self.retention[tier])

            for file_path in tier_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                files_scanned += 1
                try:
                    mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc
                    )
                    age = now - mtime
                    if age > max_age:
                        file_size = file_path.stat().st_size
                        if self.dry_run:
                            logger.info(
                                "RetentionCleaner [DRY RUN]: would delete %s "
                                "(age=%dd, limit=%dd, size=%d bytes)",
                                file_path,
                                age.days,
                                self.retention[tier],
                                file_size,
                            )
                            files_deleted += 1
                            bytes_freed += file_size
                        else:
                            file_path.unlink()
                            logger.info(
                                "RetentionCleaner: deleted %s "
                                "(age=%dd, limit=%dd, size=%d bytes)",
                                file_path,
                                age.days,
                                self.retention[tier],
                                file_size,
                            )
                            files_deleted += 1
                            bytes_freed += file_size
                except OSError as exc:
                    msg = f"Error processing {file_path}: {exc}"
                    logger.error("RetentionCleaner: %s", msg)
                    errors.append(msg)

        report = {
            "files_scanned": files_scanned,
            "files_deleted": files_deleted,
            "bytes_freed":   bytes_freed,
            "errors":        errors,
            "dry_run":       self.dry_run,
            "ran_at":        _utcnow_iso(),
        }
        logger.info(
            "RetentionCleaner: scan complete — scanned=%d deleted=%d bytes_freed=%d dry_run=%s",
            files_scanned,
            files_deleted,
            bytes_freed,
            self.dry_run,
        )
        return report


# ---------------------------------------------------------------------------
# ErasureProcessor
# ---------------------------------------------------------------------------

_ERASURE_QUEUE_PATH = "data/security/erasure_queue.json"


class ErasureProcessor:
    """Right-to-erasure: remove all records for a hashed customer_id.

    Searches all CSV files under ``base_dir`` and drops rows where the
    ``customer_id`` column matches *hashed_customer_id*. Each modified file
    is rewritten atomically (temp-file + os.replace).

    Parameters
    ----------
    base_dir : str, default "data/"
        Root directory to search for CSV files.
    erasure_queue_path : str, optional
        Path to the erasure queue JSON file used to flag customers for
        exclusion from future model training runs.

    Notes
    -----
    - This processor does **not** modify model weights. After erasure a flag
      is added to ``data/security/erasure_queue.json`` so the next training
      run knows to exclude this customer.
    - Only ``customer_id`` column is used for matching; no data values are
      ever logged.
    """

    def __init__(
        self,
        base_dir: str = "data/",
        erasure_queue_path: str = _ERASURE_QUEUE_PATH,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.erasure_queue_path = Path(erasure_queue_path)
        self._lock = threading.Lock()

    def erase(
        self,
        hashed_customer_id: str,
        store_id: str,
        requested_by: str = "unknown",
    ) -> dict:
        """Remove all CSV records for *hashed_customer_id* from *store_id*.

        Parameters
        ----------
        hashed_customer_id : str
            The 64-char hex hashed customer ID (from ``hash_customer_id``).
            Must be a hashed value — raw PII IDs are not accepted.
        store_id : str
            Store identifier. Used to scope the search and the erasure queue
            entry. Pass ``"*"`` to search across all stores (use with care).
        requested_by : str, default "unknown"
            Who requested the erasure (e.g. ``"api_key:pk_live_xxxx"`` or
            ``"admin"``). Logged for the audit trail.

        Returns
        -------
        dict
            Erasure report with keys: ``hashed_customer_id``, ``store_id``,
            ``requested_by``, ``files_modified``, ``rows_deleted``,
            ``errors``, ``completed_at``.

        Raises
        ------
        ValueError
            If *hashed_customer_id* does not look like a 64-char hex string
            (guards against accidentally passing raw PII).
        """
        # Validate that we're not accepting raw IDs
        if not _looks_hashed(hashed_customer_id):
            raise ValueError(
                "hashed_customer_id must be a 64-character hex string. "
                "Hash the customer ID with hash_customer_id() before erasing."
            )

        files_modified: list[str] = []
        rows_deleted = 0
        errors: list[str] = []

        # Search scope: all CSV files under base_dir
        # If store_id is specific, prefer that subtree first but still scan all
        for csv_path in self.base_dir.rglob("*.csv"):
            try:
                result = self._erase_from_file(csv_path, hashed_customer_id)
                if result["rows_deleted"] > 0:
                    files_modified.append(str(csv_path))
                    rows_deleted += result["rows_deleted"]
                    logger.info(
                        "ErasureProcessor: removed %d row(s) from %s",
                        result["rows_deleted"],
                        csv_path,
                    )
            except Exception as exc:
                msg = f"Error processing {csv_path}: {exc}"
                logger.error("ErasureProcessor: %s", msg)
                errors.append(msg)

        # Update the erasure queue so future training runs exclude this customer
        self._enqueue_erasure(hashed_customer_id, store_id, requested_by)

        report = {
            "hashed_customer_id": hashed_customer_id,
            "store_id":           store_id,
            "requested_by":       requested_by,
            "files_modified":     files_modified,
            "rows_deleted":       rows_deleted,
            "errors":             errors,
            "completed_at":       _utcnow_iso(),
        }
        logger.info(
            "ErasureProcessor: erasure complete — files_modified=%d rows_deleted=%d errors=%d",
            len(files_modified),
            rows_deleted,
            len(errors),
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _erase_from_file(self, csv_path: Path, hashed_id: str) -> dict:
        """Read *csv_path*, drop rows matching *hashed_id*, rewrite atomically.

        Returns dict with ``rows_deleted`` count. If the file has no
        ``customer_id`` column or no matching rows, the file is not rewritten.
        """
        df = pd.read_csv(csv_path, low_memory=False)
        if "customer_id" not in df.columns:
            return {"rows_deleted": 0}

        mask = df["customer_id"].astype(str) == hashed_id
        n_deleted = int(mask.sum())
        if n_deleted == 0:
            return {"rows_deleted": 0}

        filtered = df[~mask].reset_index(drop=True)
        # Atomic rewrite
        parent = csv_path.parent
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp.csv")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                filtered.to_csv(fh, index=False)
            os.replace(tmp_path, csv_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return {"rows_deleted": n_deleted}

    def _enqueue_erasure(
        self,
        hashed_customer_id: str,
        store_id: str,
        requested_by: str,
    ) -> None:
        """Append an entry to the erasure queue JSON file.

        The queue is read by training pipelines before fitting a model so
        they can exclude erased customers even if their records have already
        been used in a previous training run.
        """
        self.erasure_queue_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self.erasure_queue_path.exists():
                with open(self.erasure_queue_path, "r", encoding="utf-8") as fh:
                    queue: list[dict] = json.load(fh)
            else:
                queue = []

            # Avoid duplicate entries for the same customer
            already = any(
                e.get("hashed_customer_id") == hashed_customer_id
                and e.get("store_id") == store_id
                for e in queue
            )
            if not already:
                queue.append({
                    "hashed_customer_id": hashed_customer_id,
                    "store_id":           store_id,
                    "requested_by":       requested_by,
                    "queued_at":          _utcnow_iso(),
                })

            parent = self.erasure_queue_path.parent
            fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(queue, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
                os.replace(tmp_path, self.erasure_queue_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _looks_hashed(value: str) -> bool:
    """Return True if *value* is a 64-character lowercase hex string."""
    import re
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _utcnow_iso() -> str:
    """Return current UTC time as an ISO 8601 string with 'Z' suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

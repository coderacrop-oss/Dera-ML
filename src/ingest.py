"""
ingest.py — Data Ingestion Pipeline
Handles schema inference, validation, deduplication, temporal ordering,
treatment/control detection, and streaming for large files.
"""

from __future__ import annotations

import csv as _csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from data_security import filter_columns, hash_dataframe_ids
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False

_ingest_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    df: pd.DataFrame              # cleaned transaction-level data
    schema: dict                  # detected column mapping
    quality_report: dict          # validation results
    treatment_source: str         # 'embedded'|'companion'|'rfm_proxy'
    n_customers: int
    n_transactions: int
    date_range: tuple
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Column name registries for schema inference
# ---------------------------------------------------------------------------

_CUSTOMER_ID_HINTS  = {"customer_id", "cust_id", "client_id", "user_id", "customerid"}
_TIMESTAMP_HINTS    = {"timestamp", "date", "transaction_date", "order_date", "created_at",
                       "tx_date", "event_time", "datetime"}
_AMOUNT_HINTS       = {"total_price", "amount", "spend", "revenue", "sales", "price",
                       "order_value", "total_amount", "sale_amount"}
_TRANSACTION_HINTS  = {"transaction_id", "tx_id", "order_id", "invoice_id", "receipt_id"}
_TREATMENT_NAMES    = {"is_treated", "treatment", "treatment_flg", "group", "treated"}
_OUTCOME_NAMES      = {"post_promo_visit", "target", "converted", "y", "outcome",
                       "response", "purchase"}
_PRODUCT_HINTS      = {"product_name", "product", "item", "sku", "description",
                       "product_id", "item_name"}


# ---------------------------------------------------------------------------
# DataIngester
# ---------------------------------------------------------------------------

class DataIngester:
    """Data ingestion pipeline.

    Parameters
    ----------
    source_path : str
        Path to a CSV, CSV.GZ, XLSX file, or a directory of CSVs.
    chunk_size : int, default 50_000
        Number of rows per streaming chunk (used for CSV / CSV.GZ).
    """

    def __init__(
        self,
        source_path: str,
        chunk_size: int = 50_000,
        store_salt: str = "",
        store_id: str = "",
    ) -> None:
        self.source_path = str(source_path)
        self.chunk_size = chunk_size
        self.store_salt = store_salt
        self.store_id = store_id

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ingest(self) -> IngestResult:
        """Run the full ingestion pipeline and return an IngestResult."""
        warnings: list[str] = []

        raw_df = self._load(warnings)
        n_rows_raw = len(raw_df)

        # Security: drop PII columns not in the allowlist before any processing
        if _SECURITY_AVAILABLE:
            raw_df, dropped_cols = filter_columns(raw_df, warn=True)
            if dropped_cols:
                warnings.append(
                    f"Security: dropped {len(dropped_cols)} non-allowlisted "
                    f"column(s): {dropped_cols}"
                )
        else:
            _ingest_logger.warning(
                "data_security module not available — column filtering skipped."
            )

        schema = self._infer_schema(raw_df, warnings)
        clean_df, quality_stats = self._validate_and_clean(raw_df, schema, warnings)
        treatment_source = self._detect_treatment_source(schema, warnings)

        ts_col = schema.get("timestamp")
        if ts_col:
            clean_df = clean_df.sort_values(ts_col).reset_index(drop=True)

        quality_report = self._build_quality_report(n_rows_raw, clean_df, schema, quality_stats)

        # Security: hash customer_id if a store_salt is configured
        cid_col = schema.get("customer_id")
        if _SECURITY_AVAILABLE and self.store_salt and cid_col and cid_col in clean_df.columns:
            try:
                clean_df = hash_dataframe_ids(
                    clean_df,
                    id_col=cid_col,
                    store_salt=self.store_salt,
                )
                _ingest_logger.info(
                    "DataIngester: customer_id column '%s' hashed for store '%s'.",
                    cid_col,
                    self.store_id or "unknown",
                )
            except Exception as exc:
                warnings.append(f"Security: customer_id hashing failed — {exc}")
                _ingest_logger.error("DataIngester: hashing error: %s", exc)

        n_customers    = int(clean_df[cid_col].nunique()) if cid_col else 0
        n_transactions = len(clean_df)

        if ts_col and not clean_df.empty:
            ts_series  = pd.to_datetime(clean_df[ts_col], errors="coerce")
            date_range = (ts_series.min(), ts_series.max())
        else:
            date_range = (None, None)

        return IngestResult(
            df=clean_df,
            schema=schema,
            quality_report=quality_report,
            treatment_source=treatment_source,
            n_customers=n_customers,
            n_transactions=n_transactions,
            date_range=date_range,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Loading — auto-detect format
    # ------------------------------------------------------------------

    def _load(self, warnings: list[str]) -> pd.DataFrame:
        path = Path(self.source_path)

        # Directory of CSVs → concatenate
        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {path}")
            df = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)
            warnings.append(f"Concatenated {len(csv_files)} CSV files from directory.")
            return df

        suffix = "".join(path.suffixes).lower()

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="openpyxl")

        # Plain CSV or CSV.GZ — sniff delimiter from first 4 KB, then stream
        compression = "gzip" if suffix in {".csv.gz", ".gz"} else "infer"
        sep = self._sniff_sep(path, compressed=(compression == "gzip"), warnings=warnings)
        chunks = pd.read_csv(
            path,
            sep=sep,
            compression=compression,
            chunksize=self.chunk_size,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        frames = list(chunks)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _sniff_sep(path: Path, compressed: bool, warnings: list[str]) -> str:
        """Return delimiter sniffed from first 4 KB; falls back to ','."""
        import gzip
        opener = gzip.open(path, "rt", encoding="utf-8", errors="replace") if compressed \
                 else open(path, "r", encoding="utf-8", errors="replace")
        with opener as fh:
            sample = fh.read(4096)
        try:
            sep = _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except _csv.Error:
            sep = ","
        if sep != ",":
            warnings.append(f"Non-standard delimiter detected: {repr(sep)}")
        return sep

    # ------------------------------------------------------------------
    # Schema inference
    # ------------------------------------------------------------------

    def _infer_schema(self, df: pd.DataFrame, warnings: list[str]) -> dict:
        """Return a mapping of logical role → actual column name."""
        cols = list(df.columns)
        lower_map = {c.lower().strip(): c for c in cols}
        schema: dict[str, str | None] = {
            "customer_id":    None,
            "transaction_id": None,
            "timestamp":      None,
            "amount":         None,
            "treatment":      None,
            "outcome":        None,
            "product":        None,
        }

        # Exact-hint lookups (transaction_id, treatment, outcome, product)
        for role, hints in [
            ("transaction_id", _TRANSACTION_HINTS),
            ("treatment",      _TREATMENT_NAMES),
            ("outcome",        _OUTCOME_NAMES),
            ("product",        _PRODUCT_HINTS),
        ]:
            for hint in hints:
                if hint in lower_map:
                    schema[role] = lower_map[hint]
                    break

        # --- timestamp ---
        for hint in _TIMESTAMP_HINTS:
            if hint in lower_map:
                col = lower_map[hint]
                if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.5:
                    schema["timestamp"] = col
                    break
        if schema["timestamp"] is None:
            skip = {v for v in schema.values() if v}
            for col in cols:
                if col in skip:
                    continue
                try:
                    if pd.to_datetime(df[col], errors="coerce").notna().mean() > 0.6:
                        schema["timestamp"] = col
                        warnings.append(f"Timestamp inferred from column '{col}' (no hint match).")
                        break
                except Exception:
                    pass

        # --- amount ---
        for hint in _AMOUNT_HINTS:
            if hint in lower_map:
                col = lower_map[hint]
                num = pd.to_numeric(df[col], errors="coerce")
                if (num > 0).mean() > 0.5:
                    schema["amount"] = col
                    break
        if schema["amount"] is None:
            skip = {v for v in schema.values() if v}
            for col in cols:
                if col in skip:
                    continue
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notna().mean() > 0.7 and (num > 0).mean() > 0.5:
                    schema["amount"] = col
                    warnings.append(f"Amount column inferred from '{col}' (no hint match).")
                    break

        # --- customer_id ---
        for hint in _CUSTOMER_ID_HINTS:
            if hint in lower_map:
                schema["customer_id"] = lower_map[hint]
                break
        if schema["customer_id"] is None:
            skip = {v for v in schema.values() if v}
            for col in cols:
                if col in skip:
                    continue
                ratio = df[col].nunique() / max(len(df), 1)
                if 0.01 < ratio < 0.90:
                    schema["customer_id"] = col
                    warnings.append(f"customer_id inferred from '{col}' (high cardinality heuristic).")
                    break

        # --- product fallback ---
        if schema["product"] is None:
            skip = {v for v in schema.values() if v}
            for col in cols:
                if col in skip or df[col].dtype != object:
                    continue
                n_unique = df[col].nunique()
                if 2 <= n_unique <= max(200, len(df) * 0.05):
                    schema["product"] = col
                    warnings.append(f"Product column inferred from '{col}' (moderate cardinality).")
                    break

        for role, col in schema.items():
            if col is None and role not in {"product", "outcome", "treatment"}:
                warnings.append(f"Could not detect column for role '{role}'.")

        return schema

    # ------------------------------------------------------------------
    # Validation and cleaning
    # ------------------------------------------------------------------

    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        schema: dict,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, dict]:
        now = pd.Timestamp.now()
        cid_col = schema.get("customer_id")
        ts_col  = schema.get("timestamp")
        amt_col = schema.get("amount")
        tid_col = schema.get("transaction_id")

        # Null rates — one pandas call (Rung 4)
        null_rates = df.isnull().mean().round(4)
        for col, rate in null_rates.items():
            if rate > 0.30:
                warnings.append(f"Column '{col}' has {rate:.1%} nulls (>30% — potential data issue).")
            elif rate > 0.05:
                warnings.append(f"Column '{col}' has {rate:.1%} nulls (>5%).")

        stats = {
            "duplicate_ids_dropped":    0,
            "negative_amounts_dropped": 0,
            "future_ts_count":          0,
            "null_rates":               null_rates.to_dict(),
        }

        if ts_col:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            future_count = int((df[ts_col] > now).sum())
            stats["future_ts_count"] = future_count
            if future_count / max(len(df), 1) > 0.01:
                warnings.append(f"{future_count} rows ({future_count/len(df):.1%}) have future timestamps.")

        if cid_col:
            df = df.copy()
            df[cid_col] = df[cid_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            invalid_mask = df[cid_col].isin({"", "nan", "None", "NaN"})
            n_invalid = int(invalid_mask.sum())
            if n_invalid:
                warnings.append(f"Dropped {n_invalid} rows with invalid/missing customer_id.")
                df = df[~invalid_mask].copy()

        if amt_col:
            amt_num  = pd.to_numeric(df[amt_col], errors="coerce")
            neg_mask = (amt_num <= 0) | amt_num.isna()
            n_neg    = int(neg_mask.sum())
            if n_neg:
                stats["negative_amounts_dropped"] = n_neg
                warnings.append(f"Dropped {n_neg} rows with non-positive amounts.")
                df = df[~neg_mask].copy()

        if tid_col:
            n_before = len(df)
            df = df.drop_duplicates(subset=[tid_col], keep="first")
            n_dupes  = n_before - len(df)
            if n_dupes:
                stats["duplicate_ids_dropped"] = n_dupes
                warnings.append(f"Dropped {n_dupes} duplicate transaction_id rows.")

        return df.reset_index(drop=True), stats

    # ------------------------------------------------------------------
    # Treatment detection
    # ------------------------------------------------------------------

    def _detect_treatment_source(self, schema: dict, warnings: list[str]) -> str:
        if schema.get("treatment") and schema.get("outcome"):
            return "embedded"

        filename = os.path.basename(self.source_path).lower()
        if "marketing_campaign" in filename:
            base_dir = os.path.dirname(self.source_path)
            companions = [
                os.path.join(base_dir, "marketing_campaign_transactions.csv"),
                os.path.join(base_dir, "marketing_campaign_results.csv"),
            ]
            for cand in companions:
                if os.path.exists(cand):
                    warnings.append(f"Companion file detected: {cand}")
                    return "companion"
            warnings.append(
                "Filename suggests marketing_campaign but no companion file found; "
                "falling back to rfm_proxy."
            )

        return "rfm_proxy"

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------

    def _build_quality_report(
        self,
        n_rows_raw: int,
        clean_df: pd.DataFrame,
        schema: dict,
        quality_stats: dict,
    ) -> dict:
        ts_col  = schema.get("timestamp")
        trt_col = schema.get("treatment")
        out_col = schema.get("outcome")

        date_range_days = None
        if ts_col and not clean_df.empty:
            ts = pd.to_datetime(clean_df[ts_col], errors="coerce").dropna()
            if len(ts) >= 2:
                date_range_days = int((ts.max() - ts.min()).days)

        treatment_rate = (
            round(float(pd.to_numeric(clean_df[trt_col], errors="coerce").mean()), 4)
            if trt_col and trt_col in clean_df.columns else None
        )
        conversion_rate = (
            round(float(pd.to_numeric(clean_df[out_col], errors="coerce").mean()), 4)
            if out_col and out_col in clean_df.columns else None
        )

        return {
            "n_rows_raw":               n_rows_raw,
            "n_rows_clean":             len(clean_df),
            "null_rates":               quality_stats.get("null_rates", {}),
            "duplicate_ids_dropped":    quality_stats.get("duplicate_ids_dropped", 0),
            "negative_amounts_dropped": quality_stats.get("negative_amounts_dropped", 0),
            "future_ts_flagged":        quality_stats.get("future_ts_count", 0),
            "treatment_rate":           treatment_rate,
            "conversion_rate":          conversion_rate,
            "date_range_days":          date_range_days,
        }

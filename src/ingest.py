"""
ingest.py — SOTA Data Ingestion Pipeline
Handles schema inference, validation, deduplication, temporal ordering,
treatment/control detection, and streaming for large files.
"""

from __future__ import annotations

import os
import glob as _glob
import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


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
    """Production-grade data ingestion pipeline.

    Parameters
    ----------
    source_path : str
        Path to a CSV, CSV.GZ, XLSX file, or a directory of CSVs.
    chunk_size : int, default 50_000
        Number of rows per streaming chunk (used for CSV / CSV.GZ).
    """

    def __init__(self, source_path: str, chunk_size: int = 50_000) -> None:
        self.source_path = str(source_path)
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ingest(self) -> IngestResult:
        """Run the full ingestion pipeline and return an IngestResult."""
        warnings: list[str] = []

        # 1. Load raw data (streaming-friendly)
        raw_df, n_rows_raw = self._load(warnings)

        # 2. Infer schema from raw columns
        schema = self._infer_schema(raw_df, warnings)

        # 3. Streaming validation + cleaning
        clean_df, quality_stats = self._validate_and_clean(raw_df, schema, warnings)

        # 4. Treatment source detection
        treatment_source = self._detect_treatment_source(schema, warnings)

        # 5. Temporal ordering
        ts_col = schema.get("timestamp")
        if ts_col:
            clean_df = clean_df.sort_values(ts_col).reset_index(drop=True)

        # 6. Build quality report
        quality_report = self._build_quality_report(
            n_rows_raw=n_rows_raw,
            clean_df=clean_df,
            schema=schema,
            quality_stats=quality_stats,
        )

        # 7. Summary stats
        cid_col = schema.get("customer_id")
        ts_col  = schema.get("timestamp")

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

    def _load(self, warnings: list[str]) -> tuple[pd.DataFrame, int]:
        path = Path(self.source_path)

        # Directory of CSVs → concatenate
        if path.is_dir():
            csv_files = sorted(_glob.glob(str(path / "*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {path}")
            chunks: list[pd.DataFrame] = []
            for fp in csv_files:
                chunks.append(pd.read_csv(fp, low_memory=False))
            df = pd.concat(chunks, ignore_index=True)
            n_raw = len(df)
            warnings.append(f"Concatenated {len(csv_files)} CSV files from directory.")
            return df, n_raw

        suffix = "".join(path.suffixes).lower()

        # Excel
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, engine="openpyxl")
            return df, len(df)

        # CSV.GZ
        if suffix in {".csv.gz", ".gz"}:
            frames = list(self._stream_csv_gz(path, warnings))
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            return df, len(df)

        # Plain CSV (any delimiter — sniff first chunk)
        frames = list(self._stream_csv(path, warnings))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return df, len(df)

    def _stream_csv(self, path: Path, warnings: list[str]) -> Iterator[pd.DataFrame]:
        """Stream a plain CSV in chunks; sniff delimiter from first 4 KB."""
        import csv as _csv

        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            sample = fh.read(4096)

        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except _csv.Error:
            sep = ","

        if sep != ",":
            warnings.append(f"Non-standard delimiter detected: {repr(sep)}")

        reader = pd.read_csv(
            path,
            sep=sep,
            chunksize=self.chunk_size,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        for chunk in reader:
            yield chunk

    def _stream_csv_gz(self, path: Path, warnings: list[str]) -> Iterator[pd.DataFrame]:
        """Stream a gzip-compressed CSV in chunks."""
        import csv as _csv

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            sample = fh.read(4096)

        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            sep = dialect.delimiter
        except _csv.Error:
            sep = ","

        reader = pd.read_csv(
            path,
            sep=sep,
            compression="gzip",
            chunksize=self.chunk_size,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        for chunk in reader:
            yield chunk

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

        # --- transaction_id (exact name match first) ---
        for hint in _TRANSACTION_HINTS:
            if hint in lower_map:
                schema["transaction_id"] = lower_map[hint]
                break

        # --- treatment (exact name match) ---
        for name in _TREATMENT_NAMES:
            if name in lower_map:
                schema["treatment"] = lower_map[name]
                break

        # --- outcome (exact name match) ---
        for name in _OUTCOME_NAMES:
            if name in lower_map:
                schema["outcome"] = lower_map[name]
                break

        # --- product (exact name match first) ---
        for hint in _PRODUCT_HINTS:
            if hint in lower_map:
                schema["product"] = lower_map[hint]
                break

        # --- timestamp ---
        for hint in _TIMESTAMP_HINTS:
            if hint in lower_map:
                col = lower_map[hint]
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.5:
                    schema["timestamp"] = col
                    break
        if schema["timestamp"] is None:
            # Fallback: try every column
            for col in cols:
                if col in (schema["customer_id"], schema["transaction_id"]):
                    continue
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.6:
                        schema["timestamp"] = col
                        warnings.append(f"Timestamp inferred from column '{col}' (no hint match).")
                        break
                except Exception:
                    pass

        # --- amount ---
        for hint in _AMOUNT_HINTS:
            if hint in lower_map:
                col = lower_map[hint]
                numeric = pd.to_numeric(df[col], errors="coerce")
                if (numeric > 0).mean() > 0.5:
                    schema["amount"] = col
                    break
        if schema["amount"] is None:
            # Fallback: first numeric column with mostly positive values
            skip = {v for v in schema.values() if v is not None}
            for col in cols:
                if col in skip:
                    continue
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().mean() > 0.7 and (numeric > 0).mean() > 0.5:
                    schema["amount"] = col
                    warnings.append(f"Amount column inferred from '{col}' (no hint match).")
                    break

        # --- customer_id ---
        for hint in _CUSTOMER_ID_HINTS:
            if hint in lower_map:
                schema["customer_id"] = lower_map[hint]
                break
        if schema["customer_id"] is None:
            # Fallback: first high-cardinality column (not already claimed)
            skip = {v for v in schema.values() if v is not None}
            for col in cols:
                if col in skip:
                    continue
                n_unique = df[col].nunique()
                n_rows   = len(df)
                ratio    = n_unique / max(n_rows, 1)
                # "High cardinality" = >1% unique but <90% unique (not a pure key)
                if 0.01 < ratio < 0.90:
                    schema["customer_id"] = col
                    warnings.append(f"customer_id inferred from '{col}' (high cardinality heuristic).")
                    break

        # --- product fallback (moderate cardinality string) ---
        if schema["product"] is None:
            skip = {v for v in schema.values() if v is not None}
            for col in cols:
                if col in skip:
                    continue
                if df[col].dtype == object:
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= max(200, len(df) * 0.05):
                        schema["product"] = col
                        warnings.append(f"Product column inferred from '{col}' (moderate cardinality).")
                        break

        # Warn about any unmapped roles
        for role, col in schema.items():
            if col is None and role not in {"product", "outcome", "treatment"}:
                warnings.append(f"Could not detect column for role '{role}'.")

        return schema

    # ------------------------------------------------------------------
    # Streaming validation and cleaning
    # ------------------------------------------------------------------

    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        schema: dict,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, dict]:
        """Validate and clean the dataframe; return cleaned df + stats dict."""
        now = pd.Timestamp.now()

        cid_col = schema.get("customer_id")
        ts_col  = schema.get("timestamp")
        amt_col = schema.get("amount")
        tid_col = schema.get("transaction_id")

        stats = {
            "duplicate_ids_dropped": 0,
            "negative_amounts_dropped": 0,
            "future_ts_count": 0,
            "null_rates": {},
        }

        # --- Null rate per column ---
        for col in df.columns:
            null_rate = float(df[col].isna().mean())
            stats["null_rates"][col] = round(null_rate, 4)
            if null_rate > 0.30:
                warnings.append(
                    f"Column '{col}' has {null_rate:.1%} nulls (>30%% — potential data issue)."
                )
            elif null_rate > 0.05:
                warnings.append(
                    f"Column '{col}' has {null_rate:.1%} nulls (>5%%)."
                )

        # --- Parse timestamp ---
        if ts_col:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

            # Future timestamp check
            future_mask = df[ts_col] > now
            future_count = int(future_mask.sum())
            stats["future_ts_count"] = future_count
            if future_count > 0:
                future_rate = future_count / max(len(df), 1)
                if future_rate > 0.01:
                    warnings.append(
                        f"{future_count} rows ({future_rate:.1%}) have future timestamps."
                    )

        # --- Normalize customer_id to str, strip ".0" ---
        if cid_col:
            df = df.copy()
            df[cid_col] = (
                df[cid_col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            # Drop rows where customer_id is empty / 'nan'
            invalid_cid = df[cid_col].isin({"", "nan", "None", "NaN"})
            n_invalid = int(invalid_cid.sum())
            if n_invalid:
                warnings.append(f"Dropped {n_invalid} rows with invalid/missing customer_id.")
                df = df[~invalid_cid].copy()

        # --- Drop negative/zero amounts ---
        if amt_col:
            amt_numeric = pd.to_numeric(df[amt_col], errors="coerce")
            neg_mask = (amt_numeric <= 0) | amt_numeric.isna()
            n_neg = int(neg_mask.sum())
            if n_neg:
                stats["negative_amounts_dropped"] = n_neg
                warnings.append(f"Dropped {n_neg} rows with non-positive amounts.")
                df = df[~neg_mask].copy()

        # --- Deduplicate transaction IDs ---
        if tid_col:
            n_before = len(df)
            df = df.drop_duplicates(subset=[tid_col], keep="first")
            n_dupes = n_before - len(df)
            if n_dupes:
                stats["duplicate_ids_dropped"] = n_dupes
                warnings.append(f"Dropped {n_dupes} duplicate transaction_id rows.")

        return df.reset_index(drop=True), stats

    # ------------------------------------------------------------------
    # Treatment detection
    # ------------------------------------------------------------------

    def _detect_treatment_source(
        self, schema: dict, warnings: list[str]
    ) -> str:
        treatment_col = schema.get("treatment")
        outcome_col   = schema.get("outcome")
        filename      = os.path.basename(self.source_path).lower()

        if treatment_col and outcome_col:
            return "embedded"

        if "marketing_campaign" in filename:
            # Look for companion file in same directory
            base_dir = os.path.dirname(self.source_path)
            companion_candidates = [
                os.path.join(base_dir, "marketing_campaign_transactions.csv"),
                os.path.join(base_dir, "marketing_campaign_results.csv"),
            ]
            for cand in companion_candidates:
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
        n_rows_clean = len(clean_df)

        ts_col  = schema.get("timestamp")
        trt_col = schema.get("treatment")
        out_col = schema.get("outcome")

        # Date range in days
        date_range_days: int | None = None
        if ts_col and not clean_df.empty:
            ts = pd.to_datetime(clean_df[ts_col], errors="coerce").dropna()
            if len(ts) >= 2:
                date_range_days = int((ts.max() - ts.min()).days)

        # Treatment rate
        treatment_rate: float | None = None
        if trt_col and trt_col in clean_df.columns:
            trt_numeric = pd.to_numeric(clean_df[trt_col], errors="coerce")
            treatment_rate = round(float(trt_numeric.mean()), 4)

        # Conversion rate
        conversion_rate: float | None = None
        if out_col and out_col in clean_df.columns:
            out_numeric = pd.to_numeric(clean_df[out_col], errors="coerce")
            conversion_rate = round(float(out_numeric.mean()), 4)

        return {
            "n_rows_raw":               n_rows_raw,
            "n_rows_clean":             n_rows_clean,
            "null_rates":               quality_stats.get("null_rates", {}),
            "duplicate_ids_dropped":    quality_stats.get("duplicate_ids_dropped", 0),
            "negative_amounts_dropped": quality_stats.get("negative_amounts_dropped", 0),
            "future_ts_flagged":        quality_stats.get("future_ts_count", 0),
            "treatment_rate":           treatment_rate,
            "conversion_rate":          conversion_rate,
            "date_range_days":          date_range_days,
        }

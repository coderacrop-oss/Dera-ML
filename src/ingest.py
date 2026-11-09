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

import re as _re


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


# ---------------------------------------------------------------------------
# MeriPOS / DotDash helpers  (Phase 1)
# ---------------------------------------------------------------------------

# MeriPOS Transaction Report column → standard name
_MERIPOS_COLS: dict[str, str] = {
    "Bill Opens":     "timestamp",
    "Receipt number": "receipt_id",
    "Net sales":      "net_sales",
    "Gross sales":    "gross_sales",
    "Discounts":      "discount_amount",
    "Items":          "items_text",
    "Customer":       "customer_raw",
    "Payment type":   "payment_type",
    "Sales Channels": "channel",
    "Branch":         "branch",
}

# DotDash Customer Report column → standard name
_DOTDASH_COLS: dict[str, str] = {
    "กลุ่มลูกค้า (EN)":       "dotdash_segment",
    "ระดับสมาชิก":            "tier",
    "ชื่อเล่น":               "display_name",
    "เบอร์โทรศัพท์":          "phone",
    "จำนวนครั้งที่ใช้บริการ": "frequency",
    "ยอดใช้จ่าย":             "monetary",
    "ใช้บริการครั้งล่าสุด":    "last_visit",
    "แต้มที่ใช้ได้":           "points",
}

# Sentinel values that mean "no data"
_NULL_SENTINELS = {"-", "", "nan", "None", "NaN", "null"}

# Pattern: "2x Latte" or "2 x Latte" or "2×Latte"
_ITEM_QTY_RE = _re.compile(r"(\d+(?:\.\d+)?)\s*[xX×]\s*(.+?)(?=\s*,\s*\d+\s*[xX×]|$)")


def detect_source(df: pd.DataFrame) -> str:
    """Return the data source format: 'meripos', 'dotdash', or 'standard'."""
    cols = set(df.columns)
    if "Bill Opens" in cols or "Receipt number" in cols:
        return "meripos"
    if "จำนวนครั้งที่ใช้บริการ" in cols or "กลุ่มลูกค้า (EN)" in cols:
        return "dotdash"
    return "standard"


def quality_gate(df: pd.DataFrame) -> dict:
    """Assess customer-linkage rate and return mode + linked_rate.

    Returns
    -------
    dict with keys: source, mode ('full'|'partial'|'anonymous'), linked_rate, total_rows
    """
    from business_config import CONFIG
    full_thresh    = CONFIG.get("quality_gate_full_threshold",    0.30)
    partial_thresh = CONFIG.get("quality_gate_partial_threshold", 0.05)

    source = detect_source(df)

    if source == "meripos":
        col = "Customer"
    elif source == "dotdash":
        col = "เบอร์โทรศัพท์"
    else:
        col = "customer_id"

    if col not in df.columns or df.empty:
        linked_rate = 0.0
    else:
        linked = df[col].notna() & ~df[col].astype(str).str.strip().isin(_NULL_SENTINELS)
        linked_rate = float(linked.mean())

    if linked_rate >= full_thresh:
        mode = "full"
    elif linked_rate >= partial_thresh:
        mode = "partial"
    else:
        mode = "anonymous"

    return {
        "source":      source,
        "mode":        mode,
        "linked_rate": round(linked_rate, 4),
        "total_rows":  len(df),
    }


def filter_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove test/demo rows: net_sales < 10 baht and campaign names containing 'test_'.

    Logs the number of rows removed.
    """
    from business_config import CONFIG
    min_amount = CONFIG.get("test_data_min_amount", 10)

    original_len = len(df)

    # Drop tiny-amount rows (1-baht test bills)
    for col in ("Net sales", "net_sales", "total_price", "amount"):
        if col in df.columns:
            amt = pd.to_numeric(df[col], errors="coerce")
            df = df[~((amt < min_amount) & amt.notna())].copy()
            break

    # Drop rows whose string columns match "test_..." campaign pattern
    for col in df.select_dtypes(include="object").columns:
        series = df[col].astype(str)
        # skip columns that are mostly test-pattern — those are not campaign name cols
        if series.str.contains(r"(?i)\btest[_\s]", na=False, regex=True).mean() > 0.5:
            continue
        mask = series.str.contains(r"(?i)\btest[_\s]", na=False, regex=True)
        if mask.any():
            df = df[~mask].copy()

    n_removed = original_len - len(df)
    if n_removed:
        _ingest_logger.info("filter_test_data: removed %d test/demo rows", n_removed)

    return df.reset_index(drop=True)


def parse_meripos_items(df: pd.DataFrame) -> pd.DataFrame:
    """Parse MeriPOS 'Items' column into item-level rows.

    Input  : DataFrame with an 'Items' column (or 'items_text' after rename).
    Output : DataFrame — columns: receipt_id, timestamp, item_name, qty
    """
    _EMPTY = pd.DataFrame(columns=["receipt_id", "timestamp", "item_name", "qty"])

    items_col = "items_text" if "items_text" in df.columns else ("Items" if "Items" in df.columns else None)
    if items_col is None or df.empty:
        return _EMPTY

    receipt_col   = next((c for c in ("receipt_id", "Receipt number") if c in df.columns), None)
    timestamp_col = next((c for c in ("timestamp", "Bill Opens") if c in df.columns), None)

    rows: list[dict] = []
    for _, row in df.iterrows():
        receipt_id = row[receipt_col]   if receipt_col   else None
        timestamp  = row[timestamp_col] if timestamp_col else None
        items_text = str(row[items_col]) if pd.notna(row[items_col]) else ""

        matches = _ITEM_QTY_RE.findall(items_text)
        if matches:
            for qty_str, item_name in matches:
                rows.append({
                    "receipt_id": receipt_id,
                    "timestamp":  timestamp,
                    "item_name":  item_name.strip(),
                    "qty":        float(qty_str),
                })
        else:
            # Fallback: comma-separated, treat each as qty=1
            for part in items_text.split(","):
                part = part.strip()
                if part and part.lower() not in _NULL_SENTINELS:
                    rows.append({
                        "receipt_id": receipt_id,
                        "timestamp":  timestamp,
                        "item_name":  part,
                        "qty":        1.0,
                    })

    if not rows:
        return _EMPTY

    item_df = pd.DataFrame(rows)
    item_df["timestamp"] = pd.to_datetime(item_df["timestamp"], errors="coerce")
    return item_df


def ingest_meripos(path: str, filter_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalise a MeriPOS Transaction Report CSV.

    Returns
    -------
    tx_df   : normalised transaction-level DataFrame (standard columns)
    item_df : item-level DataFrame from parsed Items column
    """
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)

    if filter_test:
        raw = filter_test_data(raw)

    rename = {k: v for k, v in _MERIPOS_COLS.items() if k in raw.columns}
    df = raw.rename(columns=rename)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ("net_sales", "gross_sales", "discount_amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Derive customer_id from phone/member field where available
    if "customer_raw" in df.columns:
        linked = df["customer_raw"].notna() & ~df["customer_raw"].astype(str).str.strip().isin(_NULL_SENTINELS)
        df["customer_id"] = df["customer_raw"].where(linked, other=None)

    # Pipeline-compatibility alias
    if "net_sales" in df.columns and "total_price" not in df.columns:
        df["total_price"] = df["net_sales"]

    item_df = parse_meripos_items(df)
    return df, item_df


def ingest_dotdash(path: str) -> pd.DataFrame:
    """Load and normalise a DotDash Customer Report CSV.

    Returns a DataFrame ready for Hybrid RFM:
    customer_id, display_name, phone, frequency, monetary, recency,
    last_visit, tier, dotdash_segment, points
    """
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)

    rename = {k: v for k, v in _DOTDASH_COLS.items() if k in raw.columns}
    df = raw.rename(columns=rename)

    for col in ("frequency", "monetary"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "last_visit" in df.columns:
        df["last_visit"] = pd.to_datetime(df["last_visit"], format="%d/%m/%Y", errors="coerce")
        today = pd.Timestamp.now().normalize()
        df["recency"] = (today - df["last_visit"]).dt.days

    if "phone" in df.columns:
        df["phone"] = (
            df["phone"].astype(str)
            .str.replace(r"[-\s]", "", regex=True)
            .where(lambda s: ~s.isin(_NULL_SENTINELS), other=None)
        )

    # Placeholder customer_id — replaced by LINE userId once API is available
    df["customer_id"] = "dotdash_" + df.index.astype(str)

    return df


# ---------------------------------------------------------------------------
# AC-3.3 — Hybrid RFM Builder
# ---------------------------------------------------------------------------

def build_hybrid_rfm(
    meripos_df: pd.DataFrame,
    dotdash_df: pd.DataFrame,
    quality_mode: str = "anonymous",
) -> pd.DataFrame:
    """Combine MeriPOS transaction data and DotDash member data into a unified RFM DataFrame.

    The output format matches what ``campaign_pipeline.load_and_preprocess_csv`` returns:
    ``customer_id``, ``transaction_date``, ``amount`` — so the downstream ML pipeline
    sees a standard input regardless of which mode was used.

    Parameters
    ----------
    meripos_df : pd.DataFrame
        Normalised MeriPOS transaction DataFrame (from ingest_meripos).
        Required columns: timestamp (or Bill Opens), net_sales, customer_raw (or phone).
    dotdash_df : pd.DataFrame
        Normalised DotDash member DataFrame (from ingest_dotdash or DotDashClient).
        Required columns: customer_id, frequency, monetary, recency, last_visit.
    quality_mode : str
        'full'      — join via phone (linked_rate ≥ 0.30); RFM from actual transactions
        'partial'   — join where possible; synthesise records for unjoined DotDash members
        'anonymous' — no join; synthesise all records from DotDash member data only

    Returns
    -------
    pd.DataFrame with columns: customer_id, transaction_date, amount
        Pipeline-ready; no null customer_id or transaction_date rows.
    """
    if quality_mode == "full":
        return _hybrid_full(meripos_df, dotdash_df)
    if quality_mode == "partial":
        return _hybrid_partial(meripos_df, dotdash_df)
    # anonymous (default)
    return _hybrid_anonymous(dotdash_df)


def _hybrid_full(meripos_df: pd.DataFrame, dotdash_df: pd.DataFrame) -> pd.DataFrame:
    """Full mode: join MeriPOS transactions to DotDash members via phone.

    Matched customers get customer_id = LINE userId (or dotdash_id).
    Unmatched transactions keep 'anon_{receipt_id}' as customer_id.
    """
    tx = _meripos_to_std(meripos_df)
    if dotdash_df.empty:
        return tx

    phone_map = _build_phone_map(dotdash_df)
    if not phone_map:
        return tx

    # Map phone in tx → customer_id from DotDash
    if "phone" in tx.columns:
        mapped = tx["phone"].map(phone_map)
        mask   = mapped.notna()
        tx.loc[mask, "customer_id"] = mapped[mask]
        _ingest_logger.info(
            "build_hybrid_rfm[full]: linked %d/%d transactions via phone",
            int(mask.sum()), len(tx),
        )

    return tx[["customer_id", "transaction_date", "amount"]].dropna(
        subset=["customer_id", "transaction_date"]
    )


def _hybrid_partial(meripos_df: pd.DataFrame, dotdash_df: pd.DataFrame) -> pd.DataFrame:
    """Partial mode: join where possible, synthesise DotDash records for the rest."""
    joined = _hybrid_full(meripos_df, dotdash_df)

    # Find DotDash members not represented in joined data
    joined_cids = set(joined["customer_id"].unique())
    unjoined    = dotdash_df[~dotdash_df["customer_id"].isin(joined_cids)]

    if unjoined.empty:
        return joined

    synthetic = _synthesise_transactions(unjoined)
    result    = pd.concat([joined, synthetic], ignore_index=True)

    _ingest_logger.info(
        "build_hybrid_rfm[partial]: %d joined transactions + %d synthetic from %d unjoined members",
        len(joined), len(synthetic), len(unjoined),
    )
    return result


def _hybrid_anonymous(dotdash_df: pd.DataFrame) -> pd.DataFrame:
    """Anonymous mode: synthesise transaction records directly from DotDash member data."""
    if dotdash_df.empty:
        return pd.DataFrame(columns=["customer_id", "transaction_date", "amount"])

    result = _synthesise_transactions(dotdash_df)
    _ingest_logger.info(
        "build_hybrid_rfm[anonymous]: synthesised %d transactions from %d DotDash members",
        len(result), len(dotdash_df),
    )
    return result


# ------------------------------------------------------------------
# Private helpers for Hybrid RFM
# ------------------------------------------------------------------

def _meripos_to_std(meripos_df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard columns from normalised MeriPOS DataFrame."""
    df = meripos_df.copy()

    # Map timestamp column
    if "timestamp" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "Bill Opens" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["Bill Opens"], errors="coerce")
    else:
        df["transaction_date"] = pd.NaT

    # Amount: prefer net_sales
    if "net_sales" in df.columns:
        df["amount"] = pd.to_numeric(df["net_sales"], errors="coerce").fillna(0.0)
    elif "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    else:
        df["amount"] = 0.0

    # customer_id: phone-linked rows use phone, anonymous use receipt_id proxy
    if "customer_id" not in df.columns:
        if "phone" in df.columns:
            null_s = {"-", "", "nan", "None", "NaN", "null"}
            has_phone = df["phone"].notna() & ~df["phone"].astype(str).str.strip().isin(null_s)
            rid_col   = "receipt_id" if "receipt_id" in df.columns else None
            rid       = df[rid_col].astype(str) if rid_col else df.index.astype(str)
            df["customer_id"] = df["phone"].where(has_phone, other="anon_" + rid)
        else:
            df["customer_id"] = "anon_" + df.index.astype(str)

    # Keep phone for joining if available
    keep = ["customer_id", "transaction_date", "amount"]
    if "phone" in df.columns:
        keep.append("phone")
    return df[keep]


def _build_phone_map(dotdash_df: pd.DataFrame) -> dict[str, str]:
    """Return {normalised_phone: customer_id} from DotDash member data."""
    if "phone" not in dotdash_df.columns:
        return {}
    null_s = {"-", "", "nan", "None", "NaN", "null", "none"}
    valid  = dotdash_df[
        dotdash_df["phone"].notna()
        & ~dotdash_df["phone"].astype(str).str.strip().isin(null_s)
    ]
    return dict(zip(valid["phone"].astype(str), valid["customer_id"].astype(str)))


def _synthesise_transactions(members_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic transaction records from DotDash member F/M/R values.

    For each member with frequency N and monetary M:
    - Creates N transaction rows, each with amount = M / N
    - Last transaction date = last_visit (or today if missing)
    - Earlier transactions spread backwards at equal intervals (default 30-day gap)

    This ensures the pipeline's group-by aggregation reproduces correct F/M/R.
    """
    rows: list[dict] = []
    today = pd.Timestamp.now().normalize()

    for _, member in members_df.iterrows():
        cid       = str(member["customer_id"])
        freq      = max(1, int(member.get("frequency") or 1))
        monetary  = float(member.get("monetary") or 0.0)
        last_v    = member.get("last_visit")

        last_date = pd.to_datetime(last_v, errors="coerce") if last_v is not None else pd.NaT
        if pd.isna(last_date):
            last_date = today

        per_tx = monetary / freq if freq > 0 else 0.0

        for i in range(freq):
            # i=0 → last_visit; i=1 → 30 days earlier; etc.
            tx_date = last_date - pd.Timedelta(days=30 * i)
            rows.append({
                "customer_id":      cid,
                "transaction_date": tx_date,
                "amount":           max(0.0, per_tx),
            })

    if not rows:
        return pd.DataFrame(columns=["customer_id", "transaction_date", "amount"])

    df = pd.DataFrame(rows)
    df = df[df["amount"] > 0]
    return df.reset_index(drop=True)

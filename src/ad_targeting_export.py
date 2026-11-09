"""ad_targeting_export.py — Export Meta / LINE OA targeting audiences from campaign pipeline output.

For each IBM segment, produces 3 audience lists:
  custom_audience  — known customers to target directly (hashed email/phone)
  lookalike_seed   — best customers to seed Meta Lookalike Audience
  exclusion        — customers who should NOT see the ad (guardrail-blocked + sure-things)

Output per run:
  data/agency/ad_targeting/
    └── {run_id}/
          ├── meta_audiences.json          master JSON (all segments)
          ├── {segment}_custom.csv         hashed IDs for direct targeting
          ├── {segment}_lookalike.csv      hashed IDs for lookalike seed
          ├── {segment}_exclusion.csv      hashed IDs for exclusion
          └── manifest.json               run metadata + file list

Usage
-----
    from ad_targeting_export import export_ad_targeting

    # From campaign_planner output
    result = export_ad_targeting(
        detail_df=detail,          # output of decide_for_customers()
        crm_df=crm_df,             # CRM table: customer_id, hashed_email, hashed_phone
        output_dir="data/agency/ad_targeting",
    )

    # CLI
    python src/ad_targeting_export.py \\
        --detail-file data/processed/campaign_detail.csv \\
        --crm-file data/crm_hashed.csv \\
        --output-dir data/agency/ad_targeting
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Literal

import pandas as pd

from pipeline_logger import log


# ---------------------------------------------------------------------------
# Constants — segment → Meta audience strategy mapping
# ---------------------------------------------------------------------------

# Segments whose best use is as lookalike seed (high-value customers)
_LOOKALIKE_SEED_SEGMENTS = {
    "champions", "loyal customers", "loyal", "potential loyalist",
}

# Segments to target directly via Custom Audience
_CUSTOM_TARGET_SEGMENTS = {
    "at-risk", "hibernating", "can't lose them", "need attention",
    "about to sleep", "new customers", "promising", "churn risk",
}

# Customers who should always be excluded from paid ads
# (buying anyway = waste of budget; guardrail-blocked = not worth it)
_EXCLUSION_REASONS = {
    "sure_thing",       # cannibalization flag — would buy without promo
    "guardrail_blocked", # CFO guardrail blocked
}

# Meta's required hash format: lowercase → strip → sha256 hex
def _meta_hash(value: str) -> str:
    return hashlib.sha256(value.strip().lower().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Audience classification logic
# ---------------------------------------------------------------------------

def _classify_audiences(
    detail_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split detail_df into custom_audience, lookalike_seed, exclusion.

    Rules (applied in priority order):
    1. Exclusion   — should_send==0 (guardrail blocked) OR cannibalization_flag==1
    2. Lookalike   — should_send==1 AND segment in _LOOKALIKE_SEED_SEGMENTS
    3. Custom      — should_send==1 AND segment in _CUSTOM_TARGET_SEGMENTS
    4. Custom      — should_send==1 AND segment not matched above (safe default)
    """
    df = detail_df.copy()

    # Normalise segment column — accept either name
    seg_col = None
    for c in ("segment_name", "cluster_segment", "cluster"):
        if c in df.columns:
            seg_col = c
            break
    if seg_col is None:
        df["_seg"] = "Unknown"
        seg_col = "_seg"

    df["_seg_lower"] = df[seg_col].fillna("").str.lower().str.strip()

    # --- Exclusion ---
    excl_mask = df["should_send"] == 0
    if "cannibalization_flag" in df.columns:
        excl_mask = excl_mask | (df["cannibalization_flag"] == 1)

    exclusion   = df[excl_mask].copy()
    send_df     = df[~excl_mask & (df["should_send"] == 1)].copy()

    # --- Lookalike seed ---
    lal_mask = send_df["_seg_lower"].apply(
        lambda s: any(kw in s for kw in _LOOKALIKE_SEED_SEGMENTS)
    )
    lookalike = send_df[lal_mask].copy()
    custom    = send_df[~lal_mask].copy()

    return custom, lookalike, exclusion


# ---------------------------------------------------------------------------
# CRM matching — attach hashed identifiers
# ---------------------------------------------------------------------------

def _attach_hashed_ids(
    audience_df: pd.DataFrame,
    crm_df: pd.DataFrame | None,
    id_col: str = "customer_id",
) -> pd.DataFrame:
    """Join CRM hashed_email / hashed_phone onto audience_df.

    If crm_df is None or columns are missing, falls back to hashing customer_id
    directly (useful for LINE OA UID matching or when CRM not yet integrated).

    Meta requires pre-hashed SHA-256 lowercase email/phone.
    If the CRM provides plaintext email/phone, this function hashes them.
    If the CRM already provides hashed values, they are passed through as-is.
    """
    df = audience_df[[id_col]].drop_duplicates().copy()

    if crm_df is not None and not crm_df.empty:
        crm = crm_df.copy()
        crm[id_col] = crm[id_col].astype(str)
        df[id_col]  = df[id_col].astype(str)
        df = df.merge(crm, on=id_col, how="left")

        # Hash plaintext email/phone if not already hashed (len < 60 → plaintext)
        if "email" in df.columns:
            df["hashed_email"] = df["email"].apply(
                lambda v: _meta_hash(str(v)) if pd.notna(v) and len(str(v)) < 60 else str(v) if pd.notna(v) else None
            )
        if "phone" in df.columns:
            df["hashed_phone"] = df["phone"].apply(
                lambda v: _meta_hash(str(v)) if pd.notna(v) and len(str(v)) < 20 else str(v) if pd.notna(v) else None
            )
        # Pass-through pre-hashed columns
        for col in ("hashed_email", "hashed_phone", "line_uid"):
            if col in crm.columns and col not in df.columns:
                df[col] = df[col] if col in df.columns else None
    else:
        # Fallback: hash customer_id as placeholder
        df["hashed_customer_id"] = df[id_col].astype(str).apply(_meta_hash)

    return df


# ---------------------------------------------------------------------------
# Per-segment split
# ---------------------------------------------------------------------------

def _split_by_segment(
    df: pd.DataFrame,
    seg_col: str,
) -> dict[str, pd.DataFrame]:
    """Return {segment_name: sub_df} from a DataFrame."""
    out: dict[str, pd.DataFrame] = {}
    for seg, grp in df.groupby(seg_col):
        clean = str(seg).strip() or "Unknown"
        out[clean] = grp.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_ad_targeting(
    detail_df: pd.DataFrame,
    crm_df: pd.DataFrame | None = None,
    output_dir: str = "data/agency/ad_targeting",
    run_id: str | None = None,
    min_audience_size: int = 10,
) -> dict:
    """Export Meta/LINE OA targeting audiences from pipeline output.

    Parameters
    ----------
    detail_df        : output of decide_for_customers() — must have columns:
                       customer_id, should_send, segment_name (or cluster_segment)
    crm_df           : CRM table with customer_id + any of:
                       hashed_email, hashed_phone, line_uid, email, phone
                       Pass None to use hashed customer_id as fallback.
    output_dir       : base directory for output files
    run_id           : unique run identifier (default: timestamp)
    min_audience_size: skip writing CSV if audience has fewer than N customers
                       (Meta requires ≥100 for Custom Audience, but we keep 10
                       to allow manual review — Meta will warn at upload time)

    Returns
    -------
    dict — summary with:
        run_id, output_dir, total_custom, total_lookalike, total_exclusion,
        segments (per-segment breakdown), files (list of written file paths),
        warnings (e.g. segments below min_audience_size)
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Determine segment column
    seg_col = next(
        (c for c in ("segment_name", "cluster_segment", "cluster") if c in detail_df.columns),
        None,
    )

    # ── Step 1: Classify ─────────────────────────────────────────────────────
    custom_df, lookalike_df, exclusion_df = _classify_audiences(detail_df)

    # Preserve segment column before hashed ID join (merge may drop it)
    def _preserve_seg(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        if seg_col and seg_col in source.columns and seg_col not in target.columns:
            mapping = source[["customer_id", seg_col]].drop_duplicates("customer_id")
            target = target.merge(mapping, on="customer_id", how="left")
        return target

    # Attach hashed identifiers
    custom_df    = _preserve_seg(custom_df,    _attach_hashed_ids(custom_df,    crm_df))
    lookalike_df = _preserve_seg(lookalike_df, _attach_hashed_ids(lookalike_df, crm_df))
    exclusion_df = _preserve_seg(exclusion_df, _attach_hashed_ids(exclusion_df, crm_df))

    # ── Step 2: Per-segment breakdown ────────────────────────────────────────
    all_segments: dict[str, dict] = {}
    warnings_list: list[str] = []
    files_written: list[str] = []
    meta_audiences: list[dict] = []

    # Collect all segment names across all three lists
    all_seg_names: set[str] = set()
    for df, label in [(custom_df, "custom"), (lookalike_df, "lookalike"), (exclusion_df, "exclusion")]:
        if seg_col and seg_col in df.columns:
            all_seg_names.update(df[seg_col].fillna("Unknown").astype(str).unique())
        else:
            all_seg_names.add("All")

    # Merge all into one frame tagged by audience_type for easier grouping
    def _tag(df: pd.DataFrame, audience_type: str) -> pd.DataFrame:
        d = df.copy()
        d["_aud_type"] = audience_type
        if seg_col and seg_col in d.columns:
            d["_seg"] = d[seg_col].fillna("Unknown").astype(str)
        else:
            d["_seg"] = "All"
        return d

    tagged = pd.concat([
        _tag(custom_df,    "custom"),
        _tag(lookalike_df, "lookalike"),
        _tag(exclusion_df, "exclusion"),
    ], ignore_index=True)

    for seg in sorted(tagged["_seg"].unique()):
        seg_data: dict[str, dict] = {}
        seg_entry: dict = {"segment": seg, "custom": [], "lookalike": [], "exclusion": []}

        for aud_type in ("custom", "lookalike", "exclusion"):
            sub = tagged[(tagged["_seg"] == seg) & (tagged["_aud_type"] == aud_type)]

            # Hashed ID columns to export
            hash_cols = [c for c in ("hashed_email", "hashed_phone", "line_uid",
                                     "hashed_customer_id") if c in sub.columns]
            export_cols = ["customer_id"] + hash_cols

            n = len(sub)
            seg_data[aud_type] = {"count": n, "file": None}

            if n == 0:
                seg_entry[aud_type] = []
                continue

            if n < min_audience_size:
                warnings_list.append(
                    f"{seg} / {aud_type}: only {n} customers "
                    f"(< min_audience_size={min_audience_size}) — skipping CSV"
                )
                seg_entry[aud_type] = sub["customer_id"].tolist()
                continue

            # Write CSV
            safe_seg = seg.replace(" ", "_").replace("'", "").lower()
            fname = f"{safe_seg}_{aud_type}.csv"
            fpath = os.path.join(run_dir, fname)
            sub[export_cols].drop_duplicates("customer_id").to_csv(fpath, index=False)
            files_written.append(fpath)
            seg_data[aud_type]["file"] = fpath

            seg_entry[aud_type] = sub["customer_id"].tolist()

        all_segments[seg] = seg_data
        meta_audiences.append(seg_entry)

    # ── Step 3: Global summary CSV per audience type (written once) ──────────
    for aud_type, df in [("custom", custom_df), ("lookalike", lookalike_df), ("exclusion", exclusion_df)]:
        if df.empty:
            continue
        hash_cols = [c for c in ("hashed_email", "hashed_phone", "line_uid",
                                 "hashed_customer_id") if c in df.columns]
        out_cols = ["customer_id"] + hash_cols
        fpath = os.path.join(run_dir, f"all_{aud_type}.csv")
        if fpath not in files_written:
            df[out_cols].drop_duplicates("customer_id").to_csv(fpath, index=False)
            files_written.append(fpath)

    # ── Step 4: meta_audiences.json ──────────────────────────────────────────
    meta_json_path = os.path.join(run_dir, "meta_audiences.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id":      run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_custom":    len(custom_df),
            "total_lookalike": len(lookalike_df),
            "total_exclusion": len(exclusion_df),
            "note": (
                "custom_audience: upload to Meta Custom Audience (direct targeting). "
                "lookalike_seed: upload as source for Meta Lookalike Audience 1-3%. "
                "exclusion: upload as exclusion audience to save ad budget."
            ),
            "segments": meta_audiences,
        }, f, ensure_ascii=False, indent=2)
    files_written.append(meta_json_path)

    # ── Step 5: manifest.json ─────────────────────────────────────────────────
    manifest = {
        "run_id":          run_id,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "total_customers": len(detail_df),
        "total_custom":    len(custom_df),
        "total_lookalike": len(lookalike_df),
        "total_exclusion": len(exclusion_df),
        "match_rate_pct":  round(
            (len(custom_df) + len(lookalike_df)) / max(len(detail_df), 1) * 100, 1
        ),
        "files":    files_written,
        "warnings": warnings_list,
        "segments": {
            seg: {
                aud: {"count": v["count"], "file": os.path.basename(v["file"]) if v.get("file") else None}
                for aud, v in data.items()
            }
            for seg, data in all_segments.items()
        },
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    files_written.append(manifest_path)

    # ── Step 6: Print summary ─────────────────────────────────────────────────
    _print_summary(manifest, run_dir)

    log("ad_targeting", "export_complete",
        run_id=run_id, total_custom=manifest["total_custom"],
        total_lookalike=manifest["total_lookalike"],
        total_exclusion=manifest["total_exclusion"])

    return manifest


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_summary(manifest: dict, run_dir: str) -> None:
    W = 60
    print("\n" + "=" * W)
    print(f"  Ad Targeting Export  |  run_id: {manifest['run_id']}")
    print("=" * W)
    print(f"  Total customers      : {manifest['total_customers']:>8,}")
    print(f"  Custom Audience      : {manifest['total_custom']:>8,}  (direct target)")
    print(f"  Lookalike Seed       : {manifest['total_lookalike']:>8,}  (Meta Lookalike source)")
    print(f"  Exclusion List       : {manifest['total_exclusion']:>8,}  (save budget)")
    print(f"  Match rate           : {manifest['match_rate_pct']:>7.1f}%")
    print(f"\n  Output dir: {run_dir}")
    print("-" * W)

    for seg, data in manifest["segments"].items():
        parts = []
        for aud in ("custom", "lookalike", "exclusion"):
            n = data.get(aud, {}).get("count", 0)
            if n:
                parts.append(f"{aud}={n}")
        if parts:
            print(f"  {seg:<30} {', '.join(parts)}")

    if manifest.get("warnings"):
        print(f"\n  Warnings ({len(manifest['warnings'])}):")
        for w in manifest["warnings"]:
            print(f"    ⚠️  {w}")

    print("=" * W)
    print(f"\n  Files written: {len(manifest['files'])}")
    print(f"  meta_audiences.json + manifest.json + per-segment CSVs")
    print(f"\n  Next steps:")
    print(f"    Meta Custom Audience  → upload all_custom.csv")
    print(f"    Meta Lookalike        → upload all_lookalike.csv → create 1-3% LAL")
    print(f"    Meta Exclusion        → upload all_exclusion.csv → exclude from all ad sets")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Meta/LINE OA ad targeting audiences from campaign pipeline"
    )
    parser.add_argument("--detail-file", required=True, dest="detail_file",
                        help="CSV from decide_for_customers() output")
    parser.add_argument("--crm-file", default=None, dest="crm_file",
                        help="CRM CSV with customer_id + hashed_email/phone/line_uid (optional)")
    parser.add_argument("--output-dir", default="data/agency/ad_targeting", dest="output_dir",
                        help="Base output directory (default: data/agency/ad_targeting)")
    parser.add_argument("--run-id", default=None, dest="run_id",
                        help="Run identifier (default: timestamp)")
    parser.add_argument("--min-size", default=10, type=int, dest="min_audience_size",
                        help="Min customers per audience CSV (default: 10)")
    args = parser.parse_args()

    detail_df = pd.read_csv(args.detail_file)
    crm_df    = pd.read_csv(args.crm_file) if args.crm_file else None

    result = export_ad_targeting(
        detail_df=detail_df,
        crm_df=crm_df,
        output_dir=args.output_dir,
        run_id=args.run_id,
        min_audience_size=args.min_audience_size,
    )

    import json as _json
    print(_json.dumps(
        {k: v for k, v in result.items() if k not in ("files", "segments", "warnings")},
        ensure_ascii=False, indent=2
    ))

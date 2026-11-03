"""Peak / off-peak hours: Layer 3 computes them once; everything downstream reads the saved result."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from atomic_io import atomic_write_text

DEFAULT_DIR = "data/processed"
FILENAME = "layer3_peak_hours.json"
DEFAULT_PEAK = [11, 12, 13, 14, 15]
DEFAULT_OFF_PEAK = [8, 9, 10, 16, 17, 18]


def write_peak_hours(peak: list[int], off_peak: list[int], tz: str, threshold: float | None, source: str,
                     out_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    doc = {"generated_at": datetime.now(timezone.utc).isoformat(), "timezone": tz, "source": source,
           "threshold": None if threshold is None else round(float(threshold), 2),
           "peak_hours": sorted(int(h) for h in peak), "off_peak_hours": sorted(int(h) for h in off_peak)}
    path = os.path.join(out_dir, FILENAME)
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))
    return path


def load_peak_hours(out_dir: str = DEFAULT_DIR) -> dict | None:
    """Saved Layer 3 hours, or None if the file is missing, unreadable or malformed."""
    try:
        with open(os.path.join(out_dir, FILENAME), encoding="utf-8") as f:
            doc = json.load(f)
        for key in ("peak_hours", "off_peak_hours"):
            hours = doc[key]
            if not isinstance(hours, list) or not all(isinstance(h, int) and 0 <= h <= 23 for h in hours):
                return None
        return doc
    except (OSError, ValueError, KeyError, TypeError):
        return None


def derive_from_timestamps(timestamps: pd.Series, tz: str) -> tuple[list[int], list[int]]:
    """Same rule as Layer 3: peak = hours with volume > mean + 0.5*std, in the business timezone."""
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().dt.tz_convert(tz)
    hourly = ts.dt.hour.value_counts()
    thresh = hourly.mean() + 0.5 * hourly.std()
    return (sorted(int(h) for h in hourly[hourly > thresh].index),
            sorted(int(h) for h in hourly[hourly <= thresh].index))

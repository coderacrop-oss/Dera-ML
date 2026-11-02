"""
pipeline_logger.py — Lightweight structured logger for ml-campaign-engine.

Writes JSON-lines to output/pipeline_audit.jsonl (append mode).
Entry schema:
  {"ts": ISO8601, "layer": "L0"|"L1"|...|"L5", "event": str,
   "level": "INFO"|"WARN"|"ERROR", "data": {...}}

Usage:
  from pipeline_logger import log
  log("L1", "segmentation_complete", rows_in=5000, rows_out=4987, duration_ms=312)
  log("L4", "model_gate_failed", level="WARN", reason="below threshold")
  log("L0", "load_error", level="ERROR", exc="FileNotFoundError: ...")
"""

import json
import os
from datetime import datetime, timezone

# Resolve output path relative to this file so it works regardless of cwd
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_AUDIT_FILE = os.path.join(_OUTPUT_DIR, "pipeline_audit.jsonl")

_VALID_LEVELS = {"INFO", "WARN", "ERROR"}


def log(layer: str, event: str, level: str = "INFO", **data) -> None:
    """Append one structured JSON-line entry to pipeline_audit.jsonl.

    Args:
        layer:  Pipeline layer identifier, e.g. "L0", "L1", ..., "L5".
        event:  Short snake_case description of what happened.
        level:  Severity — "INFO", "WARN", or "ERROR" (default "INFO").
        **data: Arbitrary key/value pairs written into the "data" object.
    """
    if level not in _VALID_LEVELS:
        level = "INFO"

    entry = {
        "ts":    datetime.now(tz=timezone.utc).isoformat(),
        "layer": layer,
        "event": event,
        "level": level,
        "data":  data,
    }

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    with open(_AUDIT_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

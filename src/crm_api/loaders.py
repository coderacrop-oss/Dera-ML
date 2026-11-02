from __future__ import annotations

import re
from pathlib import Path

_SUMMARY_RE = re.compile(r"^campaign_summary_(\d{4}-(?:0[1-9]|1[0-2]))\.json$")


def list_months(data_dir: Path) -> list[str]:
    """Months with BOTH a summary json and a schedule csv, ascending."""
    months = []
    for p in Path(data_dir).glob("campaign_summary_*.json"):
        m = _SUMMARY_RE.match(p.name)
        if m and (Path(data_dir) / f"campaign_schedule_{m.group(1)}.csv").exists():
            months.append(m.group(1))
    return sorted(months)

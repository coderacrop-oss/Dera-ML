"""Layer 3 is the single source of truth for peak/off-peak hours; the scheduler must use it."""
import json

import pandas as pd
import pytest

from campaign_scheduler import detect_hours
from uplift_engine import _run_layer3 as run_layer3
from peak_hours import load_peak_hours

# transactions per Bangkok hour -> peak (mean+0.5*std) is {12, 18, 19}
WEIGHTS = {18: 60, 19: 60, 12: 40, 10: 5, 3: 2, 8: 8, 14: 6, 16: 5}


def _pos_csv(path, tz_suffix="+07:00"):
    rows, n = [], 0
    for hour, count in WEIGHTS.items():
        for i in range(count):
            n += 1
            rows.append(dict(transaction_id=f"T{n}", customer_id=f"C{n % 7}", product_name="Latte",
                             timestamp=f"2025-01-{1 + i % 5:02d} {hour:02d}:{i % 60:02d}:00{tz_suffix}",
                             quantity=1, total_price=5.0))
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_layer3_writes_peak_hours_file(workdir):
    ctx = run_layer3(_pos_csv(workdir / "pos.csv"))
    saved = json.load(open(workdir / "data/processed/layer3_peak_hours.json", encoding="utf-8"))
    assert saved["timezone"] == "Asia/Bangkok"
    assert saved["peak_hours"] == [12, 18, 19]
    assert saved["off_peak_hours"] == [3, 8, 10, 14, 16]          # observed hours that are not peak
    assert saved["source"] == "data"
    labelled = ctx.assign(h=pd.to_datetime(ctx["timestamp"], utc=True).dt.tz_convert("Asia/Bangkok").dt.hour)
    assert sorted(labelled[labelled.time_context == "Peak"].h.unique()) == saved["peak_hours"]


def test_scheduler_uses_the_hours_layer3_computed(workdir):
    run_layer3(_pos_csv(workdir / "pos.csv"))
    peak, off_peak, source = detect_hours("data/processed")
    assert (peak, off_peak) == ([12, 18, 19], [3, 8, 10, 14, 16])
    assert source == "layer3_peak_hours.json"


def test_scheduler_fallback_from_context_csv_uses_bangkok_hours_not_utc(workdir):
    run_layer3(_pos_csv(workdir / "pos.csv"))
    (workdir / "data/processed/layer3_peak_hours.json").unlink()   # only the csv is left
    peak, off_peak, source = detect_hours("data/processed")
    assert peak == [12, 18, 19]                                    # the old code returned [5, 11, 12]
    assert source == "layer3_context_flags.csv"


def test_scheduler_defaults_when_no_layer3_output(workdir):
    peak, off_peak, source = detect_hours(str(workdir / "empty"))
    assert peak == [11, 12, 13, 14, 15] and source == "defaults"


def test_corrupt_peak_hours_file_is_ignored(workdir):
    d = workdir / "data/processed"
    d.mkdir(parents=True)
    (d / "layer3_peak_hours.json").write_text("{oops", encoding="utf-8")
    assert load_peak_hours(str(d)) is None
    (d / "layer3_peak_hours.json").write_text(json.dumps({"peak_hours": "x"}), encoding="utf-8")
    assert load_peak_hours(str(d)) is None
    assert detect_hours(str(d))[2] == "defaults"

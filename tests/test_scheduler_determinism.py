import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(code: str, seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, check=True, cwd=SRC).stdout.strip()


def _fixture_csv(tmp_path) -> Path:
    actions = ["Off-Peak Traffic Driver: ส่วนลดพิเศษช่วงเงียบ", "VIP Early Access", "Standard Retention"]
    rows = [dict(customer_id=f"C{i}", segment_name="S", recommended_action=actions[i % 3],
                 offer="ลด 10%", expected_profit=10.0 + i,
                 send_window="Off-Peak Window" if i % 2 else "Peak Window",
                 channel="Email/SMS", should_send=1) for i in range(20)]
    p = tmp_path / "broadcast.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_stable_hash_is_process_independent():
    code = "from campaign_scheduler import stable_hash; print(stable_hash('C1'))"
    assert _run(code, "1") == _run(code, "2")


def test_build_schedule_identical_across_hash_seeds(tmp_path):
    p = _fixture_csv(tmp_path)
    code = (f"from campaign_scheduler import build_schedule; "
            f"s,h,_=build_schedule({str(p)!r},'2026-12',[11,12],[8,9]); "
            f"print(s['send_datetime'].astype(str).tolist(), h['customer_id'].tolist())")
    assert _run(code, "1") == _run(code, "2")


def test_atomic_write_leaves_no_temp_file(tmp_path):
    from campaign_scheduler import atomic_write_text
    target = tmp_path / "x.json"
    atomic_write_text(str(target), json.dumps({"a": "ไทย"}, ensure_ascii=False))
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": "ไทย"}

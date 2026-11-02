import json

from crm_export import write_segments_latest
from campaign_scheduler import snapshot_segments


def _payload():
    return {
        "gate_status": "PASSED",
        "target_segments": [
            {"cluster_id": 0, "segment_name": "Champions", "customer_count": 516,
             "broadcast_count": 116, "suppressed_count": 400, "recommended_promo": "VIP Early Access",
             "l4_action_breakdown": {"VIP Early Access": 116},
             "unit_economics": {"net_profit_per_head": 206.27, "total_projected_profit": 106433.28,
                                "l4_total_projected_profit": 5000.5, "financial_status": "PROFITABLE"}},
            {"cluster_id": 1, "segment_name": "At-Risk", "customer_count": 839,
             "unit_economics": {"net_profit_per_head": 10.0, "total_projected_profit": 8390.0}},
        ],
    }


def test_write_segments_latest_schema(tmp_path):
    path = write_segments_latest(_payload(), str(tmp_path))
    data = json.load(open(path, encoding="utf-8"))
    assert path.endswith("segments_latest.json")
    assert data["gate_status"] == "PASSED"
    assert data["wave_weeks"] == {"1": [1, 3], "2": [2, 4], "3": [1]}
    s0, s1 = data["segments"]
    assert (s0["id"], s0["name"], s0["customers"], s0["broadcast"], s0["suppressed"]) == (0, "Champions", 516, 116, 400)
    assert s0["action_breakdown"] == {"VIP Early Access": 116}
    assert s0["unit_economics"]["l4_total_projected_profit"] == 5000.5
    assert s1["action_breakdown"] is None and s1["broadcast"] is None and s1["recommended_promo"] is None
    assert s1["unit_economics"]["financial_status"] is None


def test_scheduler_snapshots_segments_for_month(tmp_path):
    assert snapshot_segments(str(tmp_path), "2026-12") is None  # latest missing → nothing
    write_segments_latest(_payload(), str(tmp_path))
    out = snapshot_segments(str(tmp_path), "2026-12")
    assert out.endswith("segments_2026-12.json")
    assert json.load(open(out, encoding="utf-8"))["segments"][0]["name"] == "Champions"


def test_segments_json_is_strict_json_and_tolerates_odd_values(tmp_path):
    import pandas as pd
    payload = _payload()
    payload["target_segments"][0]["unit_economics"]["net_profit_per_head"] = float("nan")
    payload["target_segments"][0]["unit_economics"]["computed_at"] = pd.Timestamp("2026-11-03")
    path = write_segments_latest(payload, str(tmp_path))
    text = open(path, encoding="utf-8").read()
    assert "NaN" not in text
    assert json.loads(text)["segments"][0]["unit_economics"]["net_profit_per_head"] is None


def test_export_segments_safe_never_raises(tmp_path):
    from crm_export import export_segments_safe
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    assert export_segments_safe(_payload(), str(blocker)) is None  # cannot mkdir under a file
    assert export_segments_safe(_payload(), str(tmp_path / "ok")).endswith("segments_latest.json")


def test_write_customer_segments_writes_every_customer_with_its_segment_name(tmp_path):
    import pandas as pd
    from crm_export import write_customer_segments
    df_rfm = pd.DataFrame({"customer_id": ["cust-0002", "007", "cust-0001"], "cluster": [1, 0, 1], "recency": [1, 2, 3]})
    segments = [{"cluster_id": 0, "segment_name": "3. At-Risk"}, {"cluster_id": 1, "segment_name": "1. Champions"}]
    path = write_customer_segments(df_rfm, segments, str(tmp_path))
    assert path.endswith("customer_segments_latest.csv")
    out = pd.read_csv(path, dtype=str)
    assert list(out.columns) == ["customer_id", "segment_id", "segment_name"]       # no rfm numbers, no extras
    assert out.customer_id.tolist() == ["007", "cust-0001", "cust-0002"]            # sorted, leading zeros kept
    assert out.set_index("customer_id").segment_name.to_dict() == {"007": "3. At-Risk", "cust-0001": "1. Champions", "cust-0002": "1. Champions"}


def test_export_customer_segments_safe_never_raises(tmp_path):
    import pandas as pd
    from crm_export import export_customer_segments_safe
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    df = pd.DataFrame({"customer_id": ["a"], "cluster": [0]})
    assert export_customer_segments_safe(df, [{"cluster_id": 0, "segment_name": "s"}], str(blocker)) is None
    assert export_customer_segments_safe(object(), [], str(tmp_path / "ok")) is None      # garbage in -> still no exception

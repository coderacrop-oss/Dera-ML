"""POST /v1/runs: a client sends transactions and a month, the API runs the pipeline and serves the result (per layer too)."""
import json
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import HDR, KEY, write_fixture_month
from crm_api.app import create_app
from crm_api.settings import Settings

CSV = "customer_id,timestamp,transaction_id,invoice_id,product_name,category,quantity,unit_price,total_price\nC1,2026-10-01T10:00:00Z,B1,B1,tea,drink,1,50,50\n"
LOG_PROXY = ("🧹 [Data Hygiene]: คัดกรองข้อมูลขยะและ Outliers ออกไป 7 รายการ\n"
             "✅ [Layer 1 Valuation] K-Means cluster จริง: k=2 (Silhouette=0.512)\n"
             "ℹ️  [Layer 4] Natural-experiment proxy (ไม่พบ real promo labels) — peak-day as treatment\n"
             "⚠️  [Layer 4] Proxy mode: uplift scores = observational CATE\n"
             "a line that is not a note\n")
LOG_REAL = LOG_PROXY.replace("Natural-experiment proxy (ไม่พบ real promo labels) — peak-day as treatment", "ใช้ Real Promotion Response labels แทน RFM proxy")


def write_layer_files(d):
    pd.DataFrame([dict(customer_id=f"C{i}", recency=10 * (i + 1), frequency=2 * (i + 1), monetary=100.0 * (i + 1), rfm_decile_segment=i + 1, kmeans_cluster=i % 2) for i in range(4)]
                 ).to_csv(d / "layer1_dual_valuation.csv", index=False)
    pd.DataFrame([dict(customer_id=f"C{i}", segment_id=i % 2, segment_name=["Champions", "Hibernating"][i % 2]) for i in range(4)]).to_csv(d / "customer_segments_latest.csv", index=False)
    (d / "segments_latest.json").write_text(json.dumps({"generated_at": "g", "gate_status": "PASSED", "segments": [
        {"id": 0, "name": "Champions", "customers": 2, "broadcast": 2, "suppressed": 0, "recommended_promo": "VIP Early Access",
         "action_breakdown": {"VIP Early Access": 2}},
        {"id": 1, "name": "Hibernating", "customers": 2, "broadcast": 0, "suppressed": 2, "recommended_promo": "ลด 15%",
         "action_breakdown": {"Do Nothing (Out of Stock)": 1, "Do Nothing (Do Not Disturb)": 1}}]}, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([dict(antecedents="tea", consequents="cake", support=0.1, confidence=0.5, lift=3.0)]).to_csv(d / "layer2_top_bundles.csv", index=False)
    (d / "layer3_peak_hours.json").write_text(json.dumps({"peak_hours": [12, 13], "off_peak_hours": [10, 11]}), encoding="utf-8")
    pd.DataFrame([dict(transaction_id="B1", customer_id="C0", product_name="tea", timestamp="t", time_context="Peak", remaining_stock=40, in_stock=True),
                  dict(transaction_id="B2", customer_id="C1", product_name="tea", timestamp="t", time_context="Peak", remaining_stock=40, in_stock=True),
                  dict(transaction_id="B3", customer_id="C1", product_name="cake", timestamp="t", time_context="Peak", remaining_stock=0, in_stock=False)]
                 ).to_csv(d / "layer3_context_flags.csv", index=False)
    pd.DataFrame([
        dict(customer_id="C0", recommended_action="VIP Early Access", should_send=1, expected_profit=10.5, guardrail_blocked=0),
        dict(customer_id="C1", recommended_action="Do Nothing (Out of Stock)", should_send=0, expected_profit=0.0, guardrail_blocked=0),
        dict(customer_id="C2", recommended_action="Do Nothing (Do Not Disturb)", should_send=0, expected_profit=0.0, guardrail_blocked=1),
        dict(customer_id="C3", recommended_action="VIP Early Access", should_send=1, expected_profit=4.5, guardrail_blocked=0)]
    ).to_csv(d / "final_campaign_engine_output.csv", index=False)


def fake_engine(log=LOG_PROXY, gate: threading.Event | None = None, fail: Exception | None = None):
    calls = []

    def engine(csv_path, month, holdout_rate, workdir):
        calls.append((csv_path.read_text(encoding="utf-8"), month, holdout_rate))
        if gate:
            assert gate.wait(10), "test gate never opened"
        if fail:
            raise fail
        out = workdir / "data" / "processed"
        out.mkdir(parents=True)
        write_fixture_month(out, month)
        write_layer_files(out)
        return log
    engine.calls = calls
    return engine


@pytest.fixture
def make_client(tmp_path):
    def make(engine, **kw):
        return TestClient(create_app(Settings(data_dir=tmp_path, api_key=KEY), run_engine=engine, **kw)), tmp_path
    return make


def wait_for(client, run_id, want=("succeeded", "failed")):
    for _ in range(200):
        body = client.get(f"/v1/runs/{run_id}", headers=HDR).json()
        if body["status"] in want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {body}")


def start(client, **over):
    return client.post("/v1/runs", headers=HDR, json={"csv": CSV, "month": "2026-12", **over})


def test_needs_the_key(make_client):
    client, _ = make_client(fake_engine())
    assert client.post("/v1/runs", json={"csv": CSV, "month": "2026-12"}).status_code == 401
    assert client.get("/v1/runs/run_x").status_code == 401
    assert client.get("/v1/runs/latest/layers").status_code == 401


def test_a_run_publishes_the_plan_and_reports_its_steps(make_client):
    engine = fake_engine()
    client, data_dir = make_client(engine)
    assert client.get("/v1/months", headers=HDR).json() == {"months": []}
    res = start(client, holdout_rate=0.1)
    assert res.status_code == 202
    body = res.json()
    assert body["run_id"].startswith("run_") and body["status"] in ("queued", "running")
    done = wait_for(client, body["run_id"])
    assert done["status"] == "succeeded" and done["error"] is None and done["finished_at"]
    assert done["label_mode"] == "proxy"
    assert engine.calls == [(CSV, "2026-12", 0.1)]
    assert client.get("/v1/months", headers=HDR).json() == {"months": ["2026-12"]}
    assert (data_dir / "layer1_dual_valuation.csv").exists()


def test_only_one_run_at_a_time(make_client):
    gate = threading.Event()
    client, _ = make_client(fake_engine(gate=gate))
    first = start(client).json()
    second = start(client)
    assert second.status_code == 409 and second.json()["error"]["code"] == "run_in_progress"
    gate.set()
    wait_for(client, first["run_id"])
    assert start(client).status_code == 202  # free again


def test_a_failed_run_leaves_the_published_plan_as_it_was(make_client):
    ok_client, data_dir = make_client(fake_engine())
    wait_for(ok_client, start(ok_client).json()["run_id"])
    before = (data_dir / "campaign_summary_2026-12.json").read_text(encoding="utf-8")
    bad = TestClient(create_app(Settings(data_dir=data_dir, api_key=KEY), run_engine=fake_engine(fail=RuntimeError("boom /secret/path"))))
    run = start(bad).json()
    done = wait_for(bad, run["run_id"])
    assert done["status"] == "failed" and done["error"] == {"code": "engine_failed"}
    assert "secret" not in json.dumps(done)
    assert (data_dir / "campaign_summary_2026-12.json").read_text(encoding="utf-8") == before


def test_refuses_bad_requests(make_client):
    client, _ = make_client(fake_engine(), max_csv_bytes=100)
    assert start(client, month="soon").status_code == 422
    assert start(client, month="2026-13").status_code == 422
    assert start(client, holdout_rate=0.5).status_code == 422
    assert start(client, csv="  ").status_code == 422
    big = start(client, csv="x" * 101)
    assert big.status_code == 413 and big.json()["error"]["code"] == "csv_too_large"
    assert client.get("/v1/runs/run_nope", headers=HDR).status_code == 404


def test_layers_before_any_run_are_404(make_client):
    client, _ = make_client(fake_engine())
    res = client.get("/v1/runs/latest/layers", headers=HDR)
    assert res.status_code == 404 and res.json()["error"]["code"] == "no_run"


def test_layers_have_the_same_shape_as_dera_ais_analysis(make_client):
    client, _ = make_client(fake_engine())
    wait_for(client, start(client).json()["run_id"])
    d = client.get("/v1/runs/latest/layers", headers=HDR).json()
    assert d["month"] == "2026-12" and d["label_mode"] == "proxy" and d["generated_at"]
    assert d["layer1"]["k"] == 2 and d["layer1"]["customers"] == 4 and d["layer1"]["silhouette"] == 0.512 and d["layer1"]["outliersRemoved"] == 7
    seg = {s["segment_name"]: s for s in d["layer1"]["segments"]}
    assert seg["Champions"] == {"segment_name": "Champions", "customer_count": 2, "avg_recency_days": 20.0, "avg_frequency": 4.0, "avg_monetary": 200.0,
                                "broadcast_count": 2, "suppressed_count": 0, "l4_action_breakdown": {"VIP Early Access": 2}}
    assert seg["Hibernating"]["l4_action_breakdown"] == {"Do Nothing (Out of Stock)": 1, "Do Nothing (Do Not Disturb)": 1}
    assert seg["Hibernating"]["broadcast_count"] == 0 and seg["Hibernating"]["suppressed_count"] == 2
    assert d["layer2"]["bundles"] == [{"a": "tea", "b": "cake", "support": 0.1, "confidence": 0.5, "lift": 3.0}]
    assert d["layer2"]["offers"][0] == {"segment": "Champions", "offer": "VIP Early Access", "basis": None}
    assert d["layer3"]["peakHours"] == [12, 13] and d["layer3"]["outOfStock"] == 1
    assert d["layer3"]["products"] == [{"product": "cake", "remaining": 0.0, "inStock": False}, {"product": "tea", "remaining": 40.0, "inStock": True}]
    assert d["layer4"] == {"total": 4, "send": 2, "suppressed": 2, "expectedProfit": 15.0, "guardrailBlocked": 1,
                           "actions": [{"action": "VIP Early Access", "count": 2}, {"action": "Do Nothing (Do Not Disturb)", "count": 1}, {"action": "Do Nothing (Out of Stock)", "count": 1}]}
    assert [w["code"] for w in d["warnings"]] == ["PROXY_LABELS"]
    assert d["notes"] == [LOG_PROXY.splitlines()[2], LOG_PROXY.splitlines()[3]]  # only the lines the engine marks as notes


def test_real_labels_give_no_proxy_warning(make_client):
    client, _ = make_client(fake_engine(log=LOG_REAL))
    wait_for(client, start(client).json()["run_id"])
    d = client.get("/v1/runs/latest/layers", headers=HDR).json()
    assert d["label_mode"] == "real" and d["warnings"] == []

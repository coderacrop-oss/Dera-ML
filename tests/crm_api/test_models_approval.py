"""Approval page + approve/reject routes for model candidates (operator tool, token protected)."""
import json
import os
import pickle

import pytest
from fastapi.testclient import TestClient

import pipeline_logger
from candidates import create_candidate, load_candidate
from crm_api.app import create_app
from crm_api.settings import Settings
from model_registry import ModelRegistry

KEY = "test-key"
META = {"approved_by": "t", "approved_at": "x", "provenance": "simulated", "metrics": None, "created_at": "x"}


def metrics(profit, sends=10, blocked=2, dn=40.0):
    return {"profit": profit, "send_count": sends, "blocked_count": blocked, "do_nothing_pct": dn}


def fake_report(compared_against=None, store_id="shop-a", label_mode="proxy", holdout_size=150):
    champ = None if compared_against is None else metrics(1000.0, 12, 3)
    return {
        "created_at": "2026-11-03T00:00:00+00:00", "compared_against_version": compared_against,
        "label_mode": label_mode, "provenance": "real-labels" if label_mode == "real" else "proxy-labels",
        "holdout_size": holdout_size, "holdout_warning": holdout_size < 100,
        "stores": [{"store_id": store_id, "rows": 500, "customers": 200, "holdout_customers": holdout_size}],
        "champion": champ, "challenger": metrics(1100.0, 14, 2),
        "per_store": {store_id: {"champion": champ, "challenger": metrics(1100.0, 14, 2)}},
        "gate": {"passed": True, "reason": "โมเดลใหม่ดีกว่าโมเดลเก่า +10.00% ผ่านเกณฑ์ 3.0% — อัปเดต champion",
                 "growth_pct": 10.0},
        "training": {"files": [{"store_id": store_id, "csv": "a.csv", "labels": None}],
                     "date_range": ["2026-01-01", "2026-03-01"], "seed": 42, "holdout_frac": 0.2},
    }


@pytest.fixture
def env(tmp_path):
    models = tmp_path / "models"
    (tmp_path / "data").mkdir()
    app = create_app(Settings(data_dir=tmp_path / "data", api_key=KEY, models_dir=models))
    return dict(client=TestClient(app), models=str(models), tmp=tmp_path)


def _new(env, **kw):
    return create_candidate(env["models"], {"n": kw.pop("n", 2)}, fake_report(**kw))


def _page(env, cid, token):
    return env["client"].get(f"/v1/models/candidates/{cid}", params={"token": token})


def _post(env, cid, action, token, approver="สมชาย"):
    return env["client"].post(f"/v1/models/candidates/{cid}/{action}", json={"token": token, "approver": approver})


def test_page_requires_valid_token(env):
    cid, token = _new(env)
    assert _page(env, cid, token).status_code == 200
    assert _page(env, cid, "wrong").status_code == 401
    assert env["client"].get(f"/v1/models/candidates/{cid}").status_code == 401
    assert _page(env, "20260101T000000Z-deadbeef", token).status_code == 404
    assert _page(env, "..", token).status_code in (404,)


def test_page_shows_all_fields(env):
    ModelRegistry(env["models"]).promote(pickle.dumps({"n": 1}), None, META)
    cid, token = _new(env, compared_against="v1")
    body = _page(env, cid, token).text
    for needle in (cid, "1,100.00", "1,000.00", "proxy", "150", "shop-a", "a.csv", "2026-03-01", "v1",
                   "โมเดลใหม่ดีกว่าโมเดลเก่า"):
        assert needle in body, needle
    assert "อนุมัติ" in body and "approver" in body


def test_proxy_mode_page_shows_caution_text(env):
    cid, token = _new(env, label_mode="proxy")
    assert "ไม่ใช่หลักฐาน" in _page(env, cid, token).text
    cid2, token2 = _new(env, label_mode="real")
    assert "ไม่ใช่หลักฐาน" not in _page(env, cid2, token2).text


def test_small_holdout_warning_is_shown(env):
    cid, token = _new(env, holdout_size=40)
    assert "น้อยกว่า 100" in _page(env, cid, token).text


def test_page_escapes_hostile_store_id_and_approver(env):
    evil = "<script>alert(1)</script>"
    cid, token = _new(env, store_id=evil)
    body = _page(env, cid, token).text
    assert evil not in body and "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert _post(env, cid, "approve", token, approver='"><img src=x onerror=y>').status_code == 200
    again = _page(env, cid, token).text
    assert "<img src=x" not in again and "&lt;img src=x" in again


def test_approve_promotes_archives_logs_and_next_serving_sees_new_version(env):
    reg = ModelRegistry(env["models"])
    reg.promote(pickle.dumps({"n": 1}), None, META)
    cid, token = _new(env, compared_against="v1", n=2)
    r = _post(env, cid, "approve", token)
    assert r.status_code == 200 and r.json() == {"status": "approved", "version": "v2"}
    assert os.path.exists(os.path.join(env["models"], "archive", "v1.pkl"))
    model, ref = reg.load_current()
    assert ref.version == "v2" and model == {"n": 2}
    assert reg.versions()[-1]["approved_by"] == "สมชาย" and reg.versions()[-1]["provenance"] == "proxy-labels"
    assert load_candidate(env["models"], cid)["status"]["status"] == "approved"
    lines = [json.loads(l) for l in open(pipeline_logger._AUDIT_FILE, encoding="utf-8")]
    assert any(l["event"] == "champion_approved" and l["data"]["candidate_id"] == cid for l in lines)


def test_first_champion_can_be_approved(env):
    cid, token = _new(env)
    assert _post(env, cid, "approve", token).json()["version"] == "v1"


def test_approve_wrong_token_changes_nothing(env):
    cid, token = _new(env)
    assert _post(env, cid, "approve", "nope").status_code == 401
    assert ModelRegistry(env["models"]).current() is None
    assert load_candidate(env["models"], cid)["status"]["status"] == "pending"


def test_approve_twice_second_is_409_not_pending(env):
    cid, token = _new(env)
    assert _post(env, cid, "approve", token).status_code == 200
    r = _post(env, cid, "approve", token)
    assert r.status_code == 409 and r.json()["error"]["code"] == "candidate_not_pending"
    assert ModelRegistry(env["models"]).current().version == "v1"


def test_two_candidates_same_base_second_approval_is_409_stale(env):
    ModelRegistry(env["models"]).promote(pickle.dumps({"n": 1}), None, META)
    c1, t1 = _new(env, compared_against="v1", n=2)
    c2, t2 = _new(env, compared_against="v1", n=3)
    assert _post(env, c1, "approve", t1).status_code == 200
    r = _post(env, c2, "approve", t2)
    assert r.status_code == 409 and r.json()["error"]["code"] == "candidate_stale"
    assert ModelRegistry(env["models"]).current().version == "v2"
    assert "ล้าสมัย" in _page(env, c2, t2).text


def test_rollback_then_old_candidate_is_stale(env):
    reg = ModelRegistry(env["models"])
    reg.promote(pickle.dumps({"n": 1}), None, META)
    reg.promote(pickle.dumps({"n": 2}), "v1", META)
    cid, token = _new(env, compared_against="v2")
    reg.rollback("v1")
    assert _post(env, cid, "approve", token).status_code == 409


def test_reject_keeps_champion(env):
    ModelRegistry(env["models"]).promote(pickle.dumps({"n": 1}), None, META)
    cid, token = _new(env, compared_against="v1")
    r = _post(env, cid, "reject", token)
    assert r.status_code == 200 and r.json() == {"status": "rejected"}
    assert ModelRegistry(env["models"]).current().version == "v1"
    assert load_candidate(env["models"], cid)["status"]["status"] == "rejected"
    assert _post(env, cid, "approve", token).status_code == 409


def test_approve_requires_approver_name(env):
    cid, token = _new(env)
    assert env["client"].post(f"/v1/models/candidates/{cid}/approve", json={"token": token}).status_code == 422
    assert _post(env, cid, "approve", token, approver="").status_code == 422
    assert _post(env, cid, "approve", token, approver="x" * 81).status_code == 422


def test_other_routes_still_require_api_key(env):
    assert env["client"].get("/v1/months").status_code == 401
    assert env["client"].get("/v1/months", headers={"X-API-Key": KEY}).status_code == 200


def test_model_routes_are_not_part_of_the_published_contract(env):
    assert not [p for p in env["client"].get("/openapi.json").json()["paths"] if "/models/" in p]


def test_page_header_names_the_version_the_numbers_were_compared_against(env):
    reg = ModelRegistry(env["models"])
    reg.promote(pickle.dumps({"n": 1}), None, META)
    c1, t1 = _new(env, compared_against="v1", n=2)
    c2, t2 = _new(env, compared_against="v1", n=3)
    assert _post(env, c1, "approve", t1).status_code == 200  # current is now v2
    assert "เทียบกับ Champion <b>v1</b>" in _page(env, c2, t2).text

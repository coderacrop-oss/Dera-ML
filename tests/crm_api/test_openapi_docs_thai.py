"""Every endpoint shown in Swagger must explain itself in Thai."""
import re

THAI = re.compile(r"[฀-๿]")
EXPECTED_ROUTES = {
    ("get", "/health"), ("get", "/v1/months"), ("get", "/v1/plans/{month}"),
    ("get", "/v1/plans/{month}/events"), ("get", "/v1/plans/{month}/calendar.ics"),
    ("get", "/v1/plans/{month}/segments"), ("get", "/v1/plans/{month}/promotions"),
    ("get", "/v1/plans/{month}/customers/{customer_id}"),
    ("get", "/v1/plans/{month}/events/{event_id}/customers"),
    ("get", "/v1/segmentation"), ("get", "/v1/segmentation/customers"), ("get", "/v1/segmentation/customers/{customer_id}"),
    ("post", "/v1/runs"), ("get", "/v1/runs/{run_id}"), ("get", "/v1/runs/latest/layers"),
    # Phase 2 — ML Campaign endpoints
    ("post", "/v1/campaign/plan"), ("post", "/v1/campaign/analyze"),
    ("get", "/v1/campaign/history"), ("post", "/v1/campaign/record-outcome"),
    ("post", "/v1/audience/export"), ("post", "/v1/content/generate"),
    ("post", "/v1/model/retrain"),
    ("post", "/v1/model/retrain-request"),
    ("get", "/v1/segment/health"),
    ("get", "/v1/send-date/recommend"),
    # Phase 5 — Execution
    ("post", "/v1/campaign/execute"),
}


def _spec(client):
    return client.get("/openapi.json").json()


def _ops(spec):
    return {(m, p): op for p, ops in spec["paths"].items() for m, op in ops.items()}


def test_exactly_the_expected_routes_are_documented(client):
    assert set(_ops(_spec(client))) == EXPECTED_ROUTES


def test_every_operation_has_thai_summary_and_description(client):
    for (method, path), op in _ops(_spec(client)).items():
        assert THAI.search(op.get("summary", "")), f"summary not Thai: {path}"
        assert len(op.get("description", "")) > 40 and THAI.search(op["description"]), f"description: {path}"


def test_every_parameter_has_thai_description(client):
    seen = set()
    for (method, path), op in _ops(_spec(client)).items():
        for prm in op.get("parameters", []):
            seen.add(prm["name"])
            assert THAI.search(prm.get("description", "")), f"parameter {prm['name']} on {path}"
    assert {"month", "customer_id", "from", "to", "wave", "event_id", "cursor", "limit", "segment_id", "run_id"} <= seen


def test_every_operation_has_a_thai_tag_and_tags_are_described(client):
    spec = _spec(client)
    declared = {t["name"]: t for t in spec.get("tags", [])}
    for (method, path), op in _ops(spec).items():
        assert op.get("tags"), path
        for tag in op["tags"]:
            assert THAI.search(tag) and THAI.search(declared[tag].get("description", "")), tag


def test_success_and_error_responses_are_described_in_thai(client):
    for (method, path), op in _ops(_spec(client)).items():
        for code, r in op["responses"].items():
            if code in ("200", "401", "404", "422", "503"):
                assert THAI.search(r.get("description", "")), f"{code} on {path}"


def test_docs_explain_what_a_wave_is_and_that_it_is_spread_over_days(client):
    spec = _spec(client)
    app_desc = spec["info"]["description"]
    assert "รอบ (wave)" in app_desc and "กระจาย" in app_desc and "ครั้งเดียว" in app_desc
    events_desc = spec["paths"]["/v1/plans/{month}/events"]["get"]["description"]
    for field in ("batch_index", "batch_total", "wave_customer_total"):
        assert field in events_desc, field


def test_security_schemes_and_app_description_are_thai(client):
    spec = _spec(client)
    for name, s in spec["components"]["securitySchemes"].items():
        assert THAI.search(s.get("description", "")), name
    assert THAI.search(spec["info"].get("description", "")) and THAI.search(spec["info"]["title"] + spec["info"]["description"])

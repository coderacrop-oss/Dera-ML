"""Every endpoint shown in Swagger must explain itself in Thai."""
import re

THAI = re.compile(r"[฀-๿]")
EXPECTED_ROUTES = {
    ("get", "/health"), ("get", "/v1/months"), ("get", "/v1/plans/{month}"),
    ("get", "/v1/plans/{month}/events"), ("get", "/v1/plans/{month}/calendar.ics"),
    ("get", "/v1/plans/{month}/segments"), ("get", "/v1/plans/{month}/promotions"),
    ("get", "/v1/plans/{month}/customers/{customer_id}"),
    ("get", "/v1/plans/{month}/events/{event_id}/customers"),
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
    assert {"month", "customer_id", "from", "to", "wave", "event_id", "cursor", "limit"} <= seen


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


def test_security_schemes_and_app_description_are_thai(client):
    spec = _spec(client)
    for name, s in spec["components"]["securitySchemes"].items():
        assert THAI.search(s.get("description", "")), name
    assert THAI.search(spec["info"].get("description", "")) and THAI.search(spec["info"]["title"] + spec["info"]["description"])

"""Swagger UI can only offer an 'Authorize' button if the spec declares the security schemes."""
from conftest import HDR


def _spec(client):
    return client.get("/openapi.json").json()


def test_spec_declares_api_key_header_and_calendar_token(client):
    schemes = _spec(client)["components"]["securitySchemes"]
    assert any(s["type"] == "apiKey" and s["in"] == "header" and s["name"] == "X-API-Key" for s in schemes.values())
    assert any(s["type"] == "apiKey" and s["in"] == "query" and s["name"] == "token" for s in schemes.values())


def test_protected_routes_require_the_key_scheme(client):
    paths = _spec(client)["paths"]
    for p in ("/v1/months", "/v1/plans/{month}", "/v1/plans/{month}/events", "/v1/plans/{month}/segments",
              "/v1/plans/{month}/promotions", "/v1/plans/{month}/customers/{customer_id}"):
        assert paths[p]["get"].get("security"), p
    assert not paths["/health"]["get"].get("security")


def test_calendar_route_accepts_key_or_token_in_spec(client):
    sec = _spec(client)["paths"]["/v1/plans/{month}/calendar.ics"]["get"]["security"]
    names = {n for req in sec for n in req}
    schemes = _spec(client)["components"]["securitySchemes"]
    kinds = {(schemes[n]["in"], schemes[n]["name"]) for n in names}
    assert kinds == {("header", "X-API-Key"), ("query", "token")}


def test_error_responses_are_documented(client):
    responses = _spec(client)["paths"]["/v1/plans/{month}"]["get"]["responses"]
    assert {"401", "404", "422", "503"} <= set(responses)
    assert "401" in _spec(client)["paths"]["/v1/months"]["get"]["responses"]


def test_auth_behaviour_unchanged(client):
    assert client.get("/v1/months").status_code == 401
    assert client.get("/v1/months", headers=HDR).status_code == 200
    assert client.get("/v1/months", headers={"X-API-Key": "wrong"}).json()["error"]["code"] == "unauthorized"

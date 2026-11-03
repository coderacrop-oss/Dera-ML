import os

import pandas as pd

from conftest import HDR, SEGMENT_NAMES, write_fixture_segmentation


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


def _pages(client, url, limit):
    out, pages, cursor = [], [], None
    while True:
        r = client.get(url + f"limit={limit}" + (f"&cursor={cursor}" if cursor else ""), headers=HDR)
        assert r.status_code == 200, r.text
        body = r.json()
        pages.append(body)
        out += body["customers"]
        cursor = body["next_cursor"]
        if not cursor:
            return out, pages


def test_segmentation_requires_the_key(data_dir, client):
    write_fixture_segmentation(data_dir)
    for path in ("/v1/segmentation", "/v1/segmentation/customers", "/v1/segmentation/customers/C001"):
        assert client.get(path).status_code == 401


def test_summary_counts_every_segment_and_all_customers(data_dir, client):
    rows = write_fixture_segmentation(data_dir)
    body = client.get("/v1/segmentation", headers=HDR).json()
    assert body["total_customers"] == len(rows)
    by_id = {s["id"]: s for s in body["segments"]}
    for sid, name in SEGMENT_NAMES.items():
        assert by_id[sid]["name"] == name
        assert by_id[sid]["customers"] == sum(1 for _, s in rows if s == sid)
    assert sum(s["customers"] for s in body["segments"]) == len(rows)
    assert body["generated_at"].endswith("+07:00")


def test_customers_are_paged_once_each_in_order_and_ids_stay_text(data_dir, client):
    rows = write_fixture_segmentation(data_dir)
    customers, pages = _pages(client, "/v1/segmentation/customers?", 5)
    ids = [c["customer_id"] for c in customers]
    assert ids == sorted(c for c, _ in rows) and len(set(ids)) == len(rows)
    assert "007" in ids and "00012" in ids                   # leading zeros survive
    assert [len(p["customers"]) for p in pages] == [5, 5, 4]
    assert pages[-1]["next_cursor"] is None and all(p["total"] == len(rows) for p in pages)
    c = next(c for c in customers if c["customer_id"] == "C001")
    assert c == {"customer_id": "C001", "segment_id": 1, "segment_name": SEGMENT_NAMES[1]}


def test_customers_can_be_filtered_by_segment(data_dir, client):
    rows = write_fixture_segmentation(data_dir)
    customers, pages = _pages(client, "/v1/segmentation/customers?segment_id=1&", 3)
    assert {c["segment_id"] for c in customers} == {1}
    assert len(customers) == sum(1 for _, s in rows if s == 1)
    assert pages[0]["total"] == len(customers)


def test_paging_parameters_are_validated(data_dir, client):
    write_fixture_segmentation(data_dir)
    for q in ("limit=0", "limit=1001", "segment_id=x", "limit=x"):
        assert client.get(f"/v1/segmentation/customers?{q}", headers=HDR).status_code == 422, q
    assert client.get("/v1/segmentation/customers?limit=1000", headers=HDR).status_code == 200


def test_bad_or_foreign_cursor_is_422_and_a_changed_file_is_409(data_dir, client):
    write_fixture_segmentation(data_dir)
    for cur in ("garbage", "e30", "!!!!"):
        r = client.get(f"/v1/segmentation/customers?cursor={cur}", headers=HDR)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_cursor", cur
    cursor = client.get("/v1/segmentation/customers?limit=2", headers=HDR).json()["next_cursor"]
    assert client.get(f"/v1/segmentation/customers?limit=2&segment_id=1&cursor={cursor}", headers=HDR).status_code == 422
    write_fixture_segmentation(data_dir, n=9)                        # the pipeline ran again
    _bump(data_dir / "customer_segments_latest.csv")
    r = client.get(f"/v1/segmentation/customers?limit=2&cursor={cursor}", headers=HDR)
    assert r.status_code == 409 and r.json()["error"]["code"] == "data_changed"
    assert client.get("/v1/segmentation/customers?limit=2", headers=HDR).status_code == 200   # restarting works


def test_one_customer_lookup(data_dir, client):
    write_fixture_segmentation(data_dir)
    r = client.get("/v1/segmentation/customers/C002", headers=HDR)
    assert r.status_code == 200 and r.json() == {"customer_id": "C002", "segment_id": 2, "segment_name": SEGMENT_NAMES[2]}
    assert client.get("/v1/segmentation/customers/007", headers=HDR).json()["segment_id"] == 1
    for cid in ("NOPE", "ลูกค้า"):
        r = client.get(f"/v1/segmentation/customers/{cid}", headers=HDR)
        assert r.status_code == 404 and r.json()["error"]["code"] == "customer_not_found", cid
    r = client.get("/v1/segmentation/customers/a%2Fb", headers=HDR)          # '/' splits the path: no such route, still a clean 404
    assert r.status_code == 404 and set(r.json()["error"]) == {"code", "message"}


def test_missing_file_is_404_and_a_corrupt_file_is_503_then_recovers(data_dir, client):
    for path in ("/v1/segmentation", "/v1/segmentation/customers", "/v1/segmentation/customers/C1"):
        r = client.get(path, headers=HDR)
        assert r.status_code == 404 and r.json()["error"]["code"] == "segmentation_not_found", path
    write_fixture_segmentation(data_dir)
    f = data_dir / "customer_segments_latest.csv"
    good = f.read_text(encoding="utf-8")
    f.write_text("customer_id,segment_id\nC1,notanumber\n", encoding="utf-8")
    _bump(f)
    assert client.get("/v1/segmentation", headers=HDR).status_code == 503
    f.write_text(good, encoding="utf-8")
    _bump(f, 20)
    assert client.get("/v1/segmentation", headers=HDR).status_code == 200


def test_etag_and_304(data_dir, client):
    write_fixture_segmentation(data_dir)
    etag = client.get("/v1/segmentation", headers=HDR).headers["ETag"]
    assert client.get("/v1/segmentation", headers={**HDR, "If-None-Match": etag}).status_code == 304


def test_empty_file_is_a_valid_empty_segmentation(data_dir, client):
    pd.DataFrame(columns=["customer_id", "segment_id", "segment_name"]).to_csv(data_dir / "customer_segments_latest.csv", index=False)
    body = client.get("/v1/segmentation", headers=HDR).json()
    assert body["total_customers"] == 0 and body["segments"] == []
    page = client.get("/v1/segmentation/customers", headers=HDR).json()
    assert page["customers"] == [] and page["next_cursor"] is None


def test_unknown_segment_ids_do_not_grow_the_server_cache(data_dir, client):
    from crm_api.segmentation import load_segmentation
    write_fixture_segmentation(data_dir)
    assert client.get("/v1/segmentation/customers?segment_id=1", headers=HDR).status_code == 200
    data = load_segmentation(data_dir)
    before = len(data._memo)
    for sid in range(1000, 1020):                                        # ids that are not segments of this shop
        r = client.get(f"/v1/segmentation/customers?segment_id={sid}", headers=HDR)
        assert r.status_code == 200 and r.json() == {"total": 0, "customers": [], "next_cursor": None}
    assert len(load_segmentation(data_dir)._memo) == before

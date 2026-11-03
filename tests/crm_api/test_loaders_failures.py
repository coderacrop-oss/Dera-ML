import os

from conftest import HDR, write_fixture_month


def _bump(path, seconds=10):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


def test_corrupt_summary_503_then_recovers(data_dir, client):
    summary = data_dir / "campaign_summary_2026-12.json"
    good = summary.read_text(encoding="utf-8")
    summary.write_text("{not json", encoding="utf-8")
    r = client.get("/v1/plans/2026-12", headers=HDR)
    assert r.status_code == 503 and r.json()["error"]["code"] == "data_unavailable"
    summary.write_text(good, encoding="utf-8")
    _bump(summary)
    assert client.get("/v1/plans/2026-12", headers=HDR).status_code == 200


def test_corrupt_schedule_503(data_dir, client):
    (data_dir / "campaign_schedule_2026-12.csv").write_bytes(b"")
    assert client.get("/v1/plans/2026-12/events", headers=HDR).status_code == 503


def test_replaced_files_are_picked_up(data_dir, client):
    first = client.get("/v1/plans/2026-12", headers=HDR)
    write_fixture_month(data_dir, schedule_rows=[])
    for f in data_dir.iterdir():
        _bump(f)
    second = client.get("/v1/plans/2026-12", headers=HDR)
    assert second.json()["totals"]["scheduled"] == 0 and second.headers["ETag"] != first.headers["ETag"]

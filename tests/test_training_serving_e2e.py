"""Whole flow: train -> approve on the page -> serve -> train again -> approve -> serve -> roll back -> serve."""
import hashlib
import json
import os
import re

import pytest
from fastapi.testclient import TestClient

import layer4_uplift_margin as l4
from crm_api.app import create_app
from crm_api.settings import Settings
from fixtures_pos import make_pos_csv
from model_registry import ModelRegistry, RegistryError
from train_champion import main


def _tree_hash(root):
    h = hashlib.sha256()
    for dp, _d, files in sorted(os.walk(root)):
        for n in sorted(files):
            if n != ".lock":
                p = os.path.join(dp, n)
                h.update(os.path.relpath(p, root).encode())
                h.update(open(p, "rb").read())
    return h.hexdigest()


def test_full_flow(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    models = str(tmp_path / "models")
    (tmp_path / "data").mkdir()
    a = make_pos_csv(tmp_path, 60, seed=1, name="a.csv")
    b = make_pos_csv(tmp_path, 60, seed=2, name="b.csv")
    args = ["--store-data", f"shop-a:{a}", "--store-data", f"shop-b:{b}", "--models-root", models]
    client = TestClient(create_app(Settings(data_dir=tmp_path / "data", api_key="k", models_dir=tmp_path / "models")))

    def serve():
        before = _tree_hash(models)
        l4.run_layer4(a, None, models_root=models)
        assert _tree_hash(models) == before, "a serving run changed the models directory"
        return json.loads((tmp_path / "data" / "processed" / "model_used.json").read_text())["champion_version"]

    def train_and_approve():
        assert main(args) == 0
        cid, token = re.search(r"/v1/models/candidates/(\S+?)\?token=(\S+)", capsys.readouterr().out).groups()
        page = client.get(f"/v1/models/candidates/{cid}", params={"token": token})
        assert page.status_code == 200 and "shop-a" in page.text and "shop-b" in page.text
        r = client.post(f"/v1/models/candidates/{cid}/approve", json={"token": token, "approver": "qa"})
        assert r.status_code == 200
        return r.json()["version"]

    with pytest.raises(RegistryError):
        serve()                                  # nothing approved yet: serving refuses, trains nothing
    assert train_and_approve() == "v1" and serve() == "v1"
    assert train_and_approve() == "v2" and serve() == "v2"
    assert main(["--rollback", "v1", "--models-root", models]) == 0
    assert serve() == "v1"
    assert [v["version"] for v in ModelRegistry(models).versions()] == ["v1", "v2"]

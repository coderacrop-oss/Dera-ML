import glob
import hashlib
import os
import pickle
import re

import pytest

import layer4_uplift_margin as l4
from candidates import candidate_model_bytes, load_candidate
from fixtures_pos import make_pos_csv
from model_registry import ModelRegistry
from train_champion import main

META = {"approved_by": "t", "approved_at": "x", "provenance": "proxy-labels", "metrics": None, "created_at": "x"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CRM_API_PUBLIC_URL", raising=False)
    a = make_pos_csv(tmp_path, 60, seed=1, name="a.csv")
    b = make_pos_csv(tmp_path, 60, seed=2, name="b.csv")
    models = str(tmp_path / "models")
    args = ["--store-data", f"shop-a:{a}", "--store-data", f"shop-b:{b}", "--models-root", models]
    return dict(tmp=tmp_path, a=a, b=b, models=models, args=args)


def _tree_hash(root):
    h = hashlib.sha256()
    for dp, _d, files in sorted(os.walk(root)):
        for n in sorted(files):
            if n != ".lock":
                p = os.path.join(dp, n)
                h.update(os.path.relpath(p, root).encode())
                h.update(open(p, "rb").read())
    return h.hexdigest()


def _candidates(models):
    return sorted(glob.glob(os.path.join(models, "candidates", "*")))


def _link(out):
    m = re.search(r"(\S+)/v1/models/candidates/(\S+?)\?token=(\S+)", out)
    assert m, out
    return m.groups()


def test_train_with_no_champion_marks_first_champion_and_prints_link(env, capsys):
    assert main(env["args"]) == 0
    base, cid, token = _link(capsys.readouterr().out)
    assert base == "http://localhost:8000"
    c = load_candidate(env["models"], cid)
    assert c["report"]["compared_against_version"] is None and c["report"]["champion"] is None
    assert "ตัวแรก" in c["report"]["gate"]["reason"]
    assert {s["store_id"] for s in c["report"]["stores"]} == {"shop-a", "shop-b"}
    assert token


def test_train_honors_public_url_and_never_touches_registry(env, capsys, monkeypatch):
    main(env["args"])
    _, cid, _ = _link(capsys.readouterr().out)
    ModelRegistry(env["models"]).promote(candidate_model_bytes(env["models"], cid), None, META)
    before = _tree_hash(os.path.join(env["models"], "registry"))
    monkeypatch.setenv("CRM_API_PUBLIC_URL", "https://ml.example.com/")
    assert main(env["args"]) == 0
    base, cid2, _ = _link(capsys.readouterr().out)
    assert base == "https://ml.example.com" and cid2 != cid
    assert _tree_hash(os.path.join(env["models"], "registry")) == before
    assert len(_candidates(env["models"])) == 2
    assert load_candidate(env["models"], cid2)["report"]["compared_against_version"] == "v1"


def test_import_legacy_then_train_compares_against_v1(env, capsys):
    ctx = l4.build_layer4_context(env["a"], None)
    model = l4._train_xlearner(ctx.raw_df[ctx.raw_df["was_treated"] == 1],
                               ctx.raw_df[ctx.raw_df["was_treated"] == 0], ctx.features)
    del model.feature_names_  # a legacy champion never recorded its feature names
    reg_dir = env["tmp"] / "models" / "registry"
    reg_dir.mkdir(parents=True)
    (reg_dir / "champion_xlearner.pkl").write_bytes(pickle.dumps(model))
    assert main(["--import-legacy", "--models-root", env["models"]]) == 0
    assert ModelRegistry(env["models"]).current().version == "v1"
    assert main(env["args"]) == 0
    _, cid, _ = _link(capsys.readouterr().out)
    r = load_candidate(env["models"], cid)["report"]
    assert r["compared_against_version"] == "v1" and r["champion"] is not None


def test_train_with_unmanifested_champion_file_tells_to_import_legacy(env, capsys):
    reg_dir = env["tmp"] / "models" / "registry"
    reg_dir.mkdir(parents=True)
    (reg_dir / "champion_xlearner.pkl").write_bytes(pickle.dumps({"n": 1}))
    assert main(env["args"]) == 1
    assert "--import-legacy" in capsys.readouterr().out
    assert _candidates(env["models"]) == []


def test_incompatible_champion_is_reported_not_compared(env, capsys):
    main(env["args"])
    _, cid, _ = _link(capsys.readouterr().out)
    other = pickle.loads(candidate_model_bytes(env["models"], cid))
    other.feature_names_ = ["something_else"]
    ModelRegistry(env["models"]).promote(pickle.dumps(other), None, META)
    assert main(env["args"]) == 0
    _, cid2, _ = _link(capsys.readouterr().out)
    r = load_candidate(env["models"], cid2)["report"]
    assert r["champion"] is None and r["compared_against_version"] == "v1"
    assert r["gate"]["passed"] is False and "ฟีเจอร์" in r["gate"]["reason"]


def test_rollback_cli_restores_version(env, capsys):
    main(env["args"])
    _, cid, _ = _link(capsys.readouterr().out)
    reg = ModelRegistry(env["models"])
    old = candidate_model_bytes(env["models"], cid)
    reg.promote(old, None, META)
    reg.promote(pickle.dumps({"n": 2}), "v1", META)
    assert main(["--rollback", "v1", "--models-root", env["models"]]) == 0
    assert reg.current().version == "v1"


def test_rollback_unknown_version_returns_1(env, capsys):
    assert main(["--rollback", "v9", "--models-root", env["models"]]) == 1
    assert "v9" in capsys.readouterr().out


def test_bad_input_returns_1_and_creates_no_candidate(env, capsys):
    assert main(["--store-data", "../x:" + env["a"], "--models-root", env["models"]]) == 1
    assert main(["--store-data", "shop-a:/nope/missing.csv", "--models-root", env["models"]]) == 1
    assert _candidates(env["models"]) == []


def test_no_mode_given_is_a_usage_error():
    with pytest.raises(SystemExit):
        main([])


def test_research_scripts_do_not_write_champion_xlearner_pkl():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    for path in glob.glob(os.path.join(src_dir, "train_*.py")):
        src = open(path, encoding="utf-8").read()
        for name in re.findall(r'(\w+)\s*=\s*[^\n]*champion_xlearner\.pkl"', src):
            assert not re.search(rf'open\(\s*{name}\s*,\s*["\']wb', src), f"{path} writes champion_xlearner.pkl via {name}"


def test_training_does_not_touch_serving_artifacts_in_cwd(env):
    d = env["tmp"] / "data" / "processed"
    d.mkdir(parents=True)
    sentinel = d / "layer3_peak_hours.json"
    sentinel.write_text("SENTINEL")
    before = sorted(os.listdir(d))
    assert main(env["args"]) == 0
    assert sentinel.read_text() == "SENTINEL" and sorted(os.listdir(d)) == before
    assert os.getcwd() == str(env["tmp"])  # cwd restored

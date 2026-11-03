import hashlib
import json
import os
import pickle

import pytest

from model_registry import ChampionRef, ModelRegistry, RegistryError, StaleCandidate

META = {"approved_by": "tester", "approved_at": "2026-11-03T00:00:00+00:00", "provenance": "real-labels",
        "metrics": {"profit": 1.0}, "created_at": "2026-11-03T00:00:00+00:00"}


def _legacy(root, obj=None):
    d = root / "registry"
    d.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(obj if obj is not None else {"n": 1})
    (d / "champion_xlearner.pkl").write_bytes(data)
    return data


def _tree_hash(root):
    """Hash of every file under root except the lock file (so a refused call can be shown to change nothing)."""
    h = hashlib.sha256()
    for dirpath, _dirs, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name == ".lock":
                continue
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


@pytest.fixture
def reg(tmp_path):
    return ModelRegistry(str(tmp_path))


def test_import_legacy_makes_v1_with_provenance_simulated(tmp_path, reg):
    _legacy(tmp_path)
    assert reg.import_legacy() == "v1"
    ref = reg.current()
    assert isinstance(ref, ChampionRef)
    assert ref.version == "v1" and ref.provenance == "simulated"
    assert reg.versions()[0]["legacy"] is True


def test_load_current_without_manifest_but_with_pickle_says_run_import_legacy(tmp_path, reg):
    _legacy(tmp_path)
    with pytest.raises(RegistryError, match="--import-legacy"):
        reg.load_current()


def test_load_current_with_nothing_raises_no_champion(reg):
    with pytest.raises(RegistryError, match="train_champion.py"):
        reg.load_current()


def test_load_current_returns_model_and_ref(tmp_path, reg):
    _legacy(tmp_path, {"n": 7})
    reg.import_legacy()
    model, ref = reg.load_current()
    assert model == {"n": 7} and ref.version == "v1"


def test_promote_archives_previous_and_replaces_atomically(tmp_path, reg):
    old = _legacy(tmp_path)
    reg.import_legacy()
    new = pickle.dumps({"n": 2})
    assert reg.promote(new, "v1", META) == "v2"
    assert (tmp_path / "archive" / "v1.pkl").read_bytes() == old
    assert (tmp_path / "registry" / "champion_xlearner.pkl").read_bytes() == new
    assert reg.current().version == "v2"
    assert reg.versions()[-1]["approved_by"] == "tester"


def test_promote_with_stale_expected_current_raises_and_changes_nothing(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    before = _tree_hash(tmp_path)
    with pytest.raises(StaleCandidate):
        reg.promote(pickle.dumps({"n": 2}), None, META)
    assert _tree_hash(tmp_path) == before


def test_second_promote_against_same_base_is_refused(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    reg.promote(pickle.dumps({"n": 2}), "v1", META)
    with pytest.raises(StaleCandidate):
        reg.promote(pickle.dumps({"n": 3}), "v1", META)


def test_first_champion_promote_with_expected_none(tmp_path, reg):
    assert reg.current() is None
    assert reg.promote(pickle.dumps({"n": 1}), None, META) == "v1"
    assert reg.load_current()[1].version == "v1"


def test_rollback_restores_old_bytes_and_manifest_current(tmp_path, reg):
    old = _legacy(tmp_path)
    reg.import_legacy()
    reg.promote(pickle.dumps({"n": 2}), "v1", META)
    reg.rollback("v1")
    assert (tmp_path / "registry" / "champion_xlearner.pkl").read_bytes() == old
    assert reg.current().version == "v1"
    assert reg.load_current()[0] == {"n": 1}
    assert (tmp_path / "archive" / "v2.pkl").exists()  # the replaced champion stays available


def test_rollback_unknown_version_raises(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    with pytest.raises(RegistryError, match="v9"):
        reg.rollback("v9")


def test_load_current_detects_modified_pickle(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    with open(tmp_path / "registry" / "champion_xlearner.pkl", "ab") as f:
        f.write(b"x")
    with pytest.raises(RegistryError, match="sha256"):
        reg.load_current()


def test_failed_replace_keeps_old_champion(tmp_path, reg, monkeypatch):
    _legacy(tmp_path)
    reg.import_legacy()
    champion = tmp_path / "registry" / "champion_xlearner.pkl"
    manifest = tmp_path / "registry" / "manifest.json"
    before = (champion.read_bytes(), manifest.read_bytes())

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        reg.promote(pickle.dumps({"n": 2}), "v1", META)
    monkeypatch.undo()
    assert (champion.read_bytes(), manifest.read_bytes()) == before
    assert not [p for p in (tmp_path / "registry").iterdir() if p.name.startswith(".tmp-")]
    assert json.loads(manifest.read_text())["current"] == "v1"
    assert reg.load_current()[0] == {"n": 1}


def test_crash_between_champion_and_manifest_is_repairable_by_rollback_of_current(tmp_path, reg, monkeypatch):
    _legacy(tmp_path)
    reg.import_legacy()

    def boom(*a, **k):
        raise OSError("crash")

    with monkeypatch.context() as m:
        m.setattr(reg, "_write_manifest", boom)
        with pytest.raises(OSError):
            reg.promote(pickle.dumps({"n": 2}), "v1", META)
    with pytest.raises(RegistryError, match="sha256"):
        reg.load_current()
    reg.rollback("v1")  # current version: restores the file the manifest describes
    assert reg.load_current()[0] == {"n": 1}


def test_crash_on_first_promote_does_not_suggest_import_legacy(tmp_path, reg, monkeypatch):
    def boom(*a, **k):
        raise OSError("crash")

    with monkeypatch.context() as m:
        m.setattr(reg, "_write_manifest", boom)
        with pytest.raises(OSError):
            reg.promote(pickle.dumps({"n": 1}), None, META)
    with pytest.raises(RegistryError) as e:
        reg.load_current()
    assert "--import-legacy" not in str(e.value)
    with pytest.raises(RegistryError):
        reg.import_legacy()  # must not adopt an unapproved candidate as v1


def test_every_version_is_archived_when_it_is_created(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    new = pickle.dumps({"n": 2})
    reg.promote(new, "v1", META)
    assert (tmp_path / "archive" / "v1.pkl").exists()
    assert (tmp_path / "archive" / "v2.pkl").read_bytes() == new


def test_missing_champion_file_under_manifest_raises_registryerror(tmp_path, reg):
    _legacy(tmp_path)
    reg.import_legacy()
    os.remove(tmp_path / "registry" / "champion_xlearner.pkl")
    with pytest.raises(RegistryError, match="missing"):
        reg.load_current()


def test_unrelated_files_in_archive_do_not_block_import_legacy(tmp_path, reg):
    """The real repo already has models/archive/champion_akeed_mock_full.pkl; only vN.pkl files count as registry archive."""
    _legacy(tmp_path)
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "champion_akeed_mock_full.pkl").write_bytes(b"old research model")
    with pytest.raises(RegistryError, match="--import-legacy"):
        reg.load_current()
    assert reg.import_legacy() == "v1"
    assert (tmp_path / "archive" / "champion_akeed_mock_full.pkl").read_bytes() == b"old research model"

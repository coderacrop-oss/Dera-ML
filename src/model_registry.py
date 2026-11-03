"""Versioned champion registry: manifest + archive, atomic writes, human-approved promotion only.

Layout under `models_root`:
  registry/champion_xlearner.pkl   the champion serving reads
  registry/manifest.json           {"current": "vN", "versions": [...]}
  registry/.lock                   flock target serialising promote/rollback/import_legacy
  archive/vN.pkl                   every superseded champion (kept for rollback)
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pickle
import re
from contextlib import contextmanager
from dataclasses import dataclass

from atomic_io import atomic_write_bytes, atomic_write_text

CHAMPION_FILE = "champion_xlearner.pkl"
_ARCHIVE_RE = re.compile(r"^v\d+\.pkl$")  # registry archive files; other files in archive/ are not ours


class RegistryError(Exception):
    pass


class StaleCandidate(RegistryError):
    pass


@dataclass(frozen=True)
class ChampionRef:
    version: str
    path: str
    sha256: str
    provenance: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _num(version: str) -> int:
    return int(version[1:])


class ModelRegistry:
    def __init__(self, models_root: str):
        self.root = os.fspath(models_root)
        self.registry_dir = os.path.join(self.root, "registry")
        self.archive_dir = os.path.join(self.root, "archive")
        self.champion_path = os.path.join(self.registry_dir, CHAMPION_FILE)
        self.manifest_path = os.path.join(self.registry_dir, "manifest.json")
        self._lock_path = os.path.join(self.registry_dir, ".lock")

    # ---- reading (never writes) -------------------------------------------------
    def _manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _entry(self, manifest: dict, version: str) -> dict | None:
        return next((v for v in manifest["versions"] if v["version"] == version), None)

    def versions(self) -> list[dict]:
        m = self._manifest()
        return list(m["versions"]) if m else []

    def current(self) -> ChampionRef | None:
        m = self._manifest()
        if not m or not m.get("current"):
            return None
        e = self._entry(m, m["current"])
        return ChampionRef(e["version"], self.champion_path, e["sha256"], e.get("provenance", "unknown"))

    def _archive_files(self) -> list:
        return [n for n in os.listdir(self.archive_dir) if _ARCHIVE_RE.fullmatch(n)] if os.path.isdir(self.archive_dir) else []

    @contextmanager
    def _shared(self):
        """Wait for a running promote/rollback (never creates the lock file: serving must not write here)."""
        if not os.path.exists(self._lock_path):
            yield
            return
        with open(self._lock_path, "r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load_current(self) -> tuple[object, ChampionRef]:
        with self._shared():
            ref = self.current()
            if ref is None:
                if os.path.exists(self.champion_path) and self._archive_files():
                    raise RegistryError("the registry is inconsistent (a champion file and archive exist but there is no "
                                        "manifest; an approval may have been interrupted). Do NOT import it as legacy; "
                                        "inspect models/registry and models/archive")
                if os.path.exists(self.champion_path):
                    raise RegistryError("a champion file exists but has no manifest; run "
                                        "`python src/train_champion.py --import-legacy` once")
                raise RegistryError("no champion model yet; train one with `python src/train_champion.py` "
                                    "and approve it")
            try:
                with open(self.champion_path, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                raise RegistryError(f"champion file {self.champion_path} is missing although the manifest says "
                                    f"{ref.version}; restore it with `python src/train_champion.py --rollback {ref.version}`")
        if _sha256(data) != ref.sha256:
            raise RegistryError(f"champion file {self.champion_path} does not match the manifest sha256 "
                                f"for {ref.version}; refusing to serve (repair: `--rollback {ref.version}`)")
        return pickle.loads(data), ref

    # ---- writing ----------------------------------------------------------------
    @contextmanager
    def _locked(self):
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        with open(self._lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write_manifest(self, manifest: dict) -> None:
        atomic_write_text(self.manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))

    def _archive_bytes(self, version: str, data: bytes) -> None:
        dest = os.path.join(self.archive_dir, f"{version}.pkl")
        if not os.path.exists(dest):
            atomic_write_bytes(dest, data)

    def _archive_current(self, manifest: dict) -> None:
        cur = manifest.get("current")
        if not cur:
            return
        with open(self.champion_path, "rb") as f:
            data = f.read()
        if _sha256(data) != self._entry(manifest, cur)["sha256"]:
            raise RegistryError(f"champion file does not match the manifest sha256 for {cur}; "
                                f"repair it first with `--rollback {cur}`")
        self._archive_bytes(cur, data)

    def import_legacy(self) -> str:
        with self._locked():
            if self._manifest() is not None:
                raise RegistryError("manifest already exists; nothing to import")
            if self._archive_files():
                raise RegistryError("an archive exists but there is no manifest; refusing to adopt the champion file "
                                    "as legacy v1 (an approval may have been interrupted). Inspect models/ first")
            if not os.path.exists(self.champion_path):
                raise RegistryError(f"no legacy champion at {self.champion_path}")
            with open(self.champion_path, "rb") as f:
                data = f.read()
            self._archive_bytes("v1", data)
            self._write_manifest({"current": "v1", "versions": [{
                "version": "v1", "sha256": _sha256(data), "created_at": None, "approved_by": None,
                "approved_at": None, "provenance": "simulated", "metrics": None, "legacy": True}]})
            return "v1"

    def promote(self, model_bytes: bytes, expected_current: str | None, meta: dict) -> str:
        with self._locked():
            manifest = self._manifest() or {"current": None, "versions": []}
            if manifest["current"] != expected_current:
                raise StaleCandidate(f"candidate was compared against {expected_current!r} but the current "
                                     f"champion is {manifest['current']!r}")
            version = f"v{max([_num(v['version']) for v in manifest['versions']] or [0]) + 1}"
            self._archive_current(manifest)
            self._archive_bytes(version, model_bytes)          # every version is archived when it is created
            atomic_write_bytes(self.champion_path, model_bytes)
            manifest["versions"].append({
                "version": version, "sha256": _sha256(model_bytes), "created_at": meta.get("created_at"),
                "approved_by": meta.get("approved_by"), "approved_at": meta.get("approved_at"),
                "provenance": meta.get("provenance", "unknown"), "metrics": meta.get("metrics"),
                "legacy": False})
            manifest["current"] = version
            self._write_manifest(manifest)
            return version

    def _restore_from_archive(self, version: str, entry: dict) -> None:
        src = os.path.join(self.archive_dir, f"{version}.pkl")
        if not os.path.exists(src):
            raise RegistryError(f"archive for {version} is missing")
        with open(src, "rb") as f:
            data = f.read()
        if _sha256(data) != entry["sha256"]:
            raise RegistryError(f"archive for {version} does not match its manifest sha256")
        atomic_write_bytes(self.champion_path, data)

    def rollback(self, version: str) -> None:
        """Make `version` current. For the current version this repairs a champion file that is missing or
        does not match the manifest (e.g. after an interrupted approval)."""
        with self._locked():
            manifest = self._manifest()
            entry = self._entry(manifest, version) if manifest else None
            if entry is None:
                raise RegistryError(f"unknown version {version!r}")
            if manifest["current"] == version:
                ok = os.path.exists(self.champion_path)
                if ok:
                    with open(self.champion_path, "rb") as f:
                        ok = _sha256(f.read()) == entry["sha256"]
                if not ok:
                    self._restore_from_archive(version, entry)
                return
            self._archive_current(manifest)
            self._restore_from_archive(version, entry)
            manifest["current"] = version
            self._write_manifest(manifest)

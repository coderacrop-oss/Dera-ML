"""Write side of the CRM API: a client posts transactions, the API runs the pipeline on them and serves the result.

One run at a time. The pipeline works in its own temporary directory; only a SUCCESSFUL run replaces files in `data_dir`
(file by file with an atomic rename, the month's summary last), so a failed run leaves the published plan as it was.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from fastapi import Path as PathParam
from pydantic import BaseModel, Field, field_validator

from . import docs
from .errors import ApiError
from .settings import Settings

SRC = Path(__file__).resolve().parents[1]
DEFAULT_MAX_CSV_BYTES = 20 * 1024 * 1024
LATEST_LAYERS = "latest_run_layers.json"
PIPELINE_TIMEOUT_S = 900
NOTE_MARKERS = ("⚠️", "คำเตือน", "💡", "ℹ️")
PROXY_WARNING_TH = "ไม่มีประวัติโปรที่ใช้ได้ ผลส่วน 4 เป็นการเดา ไม่ใช่ผลของโปรจริง"

Engine = Callable[[Path, str, float, Path], str]
"""engine(csv_path, month, holdout_rate, workdir) -> the pipeline's log text; leaves its files in workdir/data/processed."""


class EngineFailed(Exception):
    pass


def subprocess_engine(csv_path: Path, month: str, holdout_rate: float, workdir: Path) -> str:
    """The real thing: `campaign_pipeline.py <csv>` then `campaign_scheduler.py --month` inside `workdir` (its data/processed is where they write)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    def run(cmd: list[str]) -> str:
        try:
            done = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8", timeout=PIPELINE_TIMEOUT_S, env=env)
        except subprocess.TimeoutExpired as exc:
            raise EngineFailed("timeout") from exc
        if done.returncode != 0:
            raise EngineFailed(f"exit {done.returncode}")
        return done.stdout

    log = run([sys.executable, str(SRC / "campaign_pipeline.py"), str(csv_path)])
    run([sys.executable, str(SRC / "campaign_scheduler.py"), "--month", month, "--holdout", str(holdout_rate)])
    return log


class RunRequest(BaseModel):
    csv: str
    month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    holdout_rate: float = Field(0.05, ge=0.0, le=0.2)

    @field_validator("csv")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("csv must not be empty")
        return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw)


def build_layers(directory: Path, log: str, month: str, generated_at: str) -> dict:
    """Layers 1-4 of a finished run, in the shape of Dera AI's `plans/<month>/analysis`, from the files the pipeline left."""
    rfm = _read(directory / "layer1_dual_valuation.csv")
    seg_of = _read(directory / "customer_segments_latest.csv")
    merged = seg_of.merge(rfm[["customer_id", "recency", "frequency", "monetary"]], on="customer_id", how="left")
    segments = [{"segment_name": str(name), "customer_count": int(len(g)), "avg_recency_days": round(float(g["recency"].mean()), 2),
                 "avg_frequency": round(float(g["frequency"].mean()), 2), "avg_monetary": round(float(g["monetary"].mean()), 2)}
                for name, g in merged.groupby("segment_name", sort=True)]
    sil = re.search(r"Silhouette=([0-9.]+)", log)
    outl = re.search(r"ออกไป (\d+) รายการ", log)
    layer1 = {"k": int(rfm["kmeans_cluster"].nunique()), "silhouette": float(sil.group(1)) if sil else None, "customers": int(len(rfm)),
              "outliersRemoved": int(outl.group(1)) if outl else None, "segments": segments}

    bundles = [{"a": str(r.antecedents), "b": str(r.consequents), "support": float(r.support), "confidence": float(r.confidence), "lift": float(r.lift)}
               for r in _read(directory / "layer2_top_bundles.csv").itertuples()]
    promos = json.loads((directory / "segments_latest.json").read_text(encoding="utf-8")).get("segments", [])
    offers = [{"segment": str(s.get("name")), "offer": str(s.get("recommended_promo")), "basis": None} for s in promos if s.get("recommended_promo")]
    layer2 = {"bundles": bundles, "offers": offers}

    final = _read(directory / "final_campaign_engine_output.csv")
    flags = _read(directory / "layer3_context_flags.csv").drop_duplicates("product_name")
    products = [{"product": str(r.product_name), "remaining": None if pd.isna(r.remaining_stock) else float(r.remaining_stock), "inStock": bool(r.in_stock)} for r in flags.itertuples()]
    products.sort(key=lambda p: (p["inStock"], p["remaining"] if p["remaining"] is not None else 0, p["product"]))
    peak = json.loads((directory / "layer3_peak_hours.json").read_text(encoding="utf-8")).get("peak_hours", [])
    layer3 = {"peakHours": [int(h) for h in peak], "products": products, "outOfStock": int((final["recommended_action"] == "Do Nothing (Out of Stock)").sum())}

    counts = final["recommended_action"].value_counts().to_dict()
    sent = final["should_send"] == 1
    layer4 = {"total": int(len(final)), "send": int(sent.sum()), "suppressed": int((~sent).sum()),
              "expectedProfit": round(float(final.loc[sent, "expected_profit"].sum()), 2), "guardrailBlocked": int(final["guardrail_blocked"].sum()),
              "actions": [{"action": str(a), "count": int(n)} for a, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]}

    label_mode = "proxy" if "Natural-experiment proxy" in log else "real" if "Real Promotion Response" in log else None
    return {"month": month, "generated_at": generated_at, "label_mode": label_mode, "layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4,
            "warnings": [{"code": "PROXY_LABELS", "message_th": PROXY_WARNING_TH}] if label_mode == "proxy" else [],
            "notes": [ln.strip() for ln in log.splitlines() if ln.strip().startswith(NOTE_MARKERS)][:12]}


def _publish(source: Path, data_dir: Path, run_id: str) -> None:
    """Copy every file of `source` into `data_dir` (atomic rename per file); a month's summary goes last, so the month only appears when whole."""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in source.iterdir() if p.is_file()), key=lambda p: (p.name.startswith("campaign_summary_"), p.name))
    for f in files:
        tmp = data_dir / f".{f.name}.{run_id}.tmp"
        shutil.copyfile(f, tmp)
        os.replace(tmp, data_dir / f.name)


class RunManager:
    def __init__(self, settings: Settings, engine: Engine, max_csv_bytes: int):
        self.settings, self.engine, self.max_csv_bytes = settings, engine, max_csv_bytes
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}
        self._running: str | None = None

    def start(self, req: RunRequest) -> dict:
        if len(req.csv.encode("utf-8")) > self.max_csv_bytes:
            raise ApiError(413, "csv_too_large", f"csv is larger than {self.max_csv_bytes} bytes")
        with self._lock:
            if self._running is not None:
                raise ApiError(409, "run_in_progress", "a run is already in progress")
            run_id = f"run_{uuid.uuid4().hex[:16]}"
            self._runs[run_id] = {"run_id": run_id, "status": "queued", "step": None, "month": req.month, "started_at": None, "finished_at": None, "label_mode": None, "error": None}
            self._running = run_id
        threading.Thread(target=self._work, args=(run_id, req), daemon=True).start()
        return dict(self._runs[run_id])

    def get(self, run_id: str) -> dict:
        run = self._runs.get(run_id)
        if run is None:
            raise ApiError(404, "run_not_found", "no such run")
        return dict(run)

    def latest_layers(self) -> dict:
        path = Path(self.settings.data_dir) / LATEST_LAYERS
        if not path.exists():
            raise ApiError(404, "no_run", "no run has finished yet")
        return json.loads(path.read_text(encoding="utf-8"))

    def _work(self, run_id: str, req: RunRequest) -> None:
        run = self._runs[run_id]
        workdir = Path(tempfile.mkdtemp(prefix="crm-run-"))
        try:
            run.update(status="running", step="layers", started_at=_now())
            csv_path = workdir / "input.csv"
            csv_path.write_text(req.csv, encoding="utf-8")
            log = self.engine(csv_path, req.month, req.holdout_rate, workdir)
            produced = workdir / "data" / "processed"
            run["step"] = "publishing"
            layers = build_layers(produced, log, req.month, _now())
            (produced / LATEST_LAYERS).write_text(json.dumps(layers, ensure_ascii=False), encoding="utf-8")
            _publish(produced, Path(self.settings.data_dir), run_id)
            run.update(status="succeeded", label_mode=layers["label_mode"])
        except Exception:  # the cause stays out of the answer (paths, customer ids); the code says which part failed
            run.update(status="failed", error={"code": "engine_failed"})
        finally:
            run["finished_at"] = _now()
            shutil.rmtree(workdir, ignore_errors=True)
            with self._lock:
                self._running = None


router = APIRouter()


def _manager(request: Request) -> RunManager:
    return request.app.state.runs


@router.post("/runs", status_code=202, responses=docs.ERR_RUNS_START, **docs.RUNS_START)
def start_run(request: Request, body: RunRequest):
    run = _manager(request).start(body)
    return {"run_id": run["run_id"], "status": run["status"]}


@router.get("/runs/latest/layers", **docs.RUNS_LAYERS)
def latest_layers(request: Request):
    return _manager(request).latest_layers()


@router.get("/runs/{run_id}", **docs.RUNS_STATUS)
def get_run(request: Request, run_id: str = PathParam(..., description=docs.P_RUN_ID)):
    return _manager(request).get(run_id)

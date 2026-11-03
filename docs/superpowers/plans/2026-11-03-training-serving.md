# Training / Serving Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serving runs only score with an approved champion; a separate training command builds a candidate from pooled shop data, compares it with the champion on held-out customers, and a person approves it on a web page before it becomes champion.

**Architecture:** A versioned `ModelRegistry` (manifest + archive, atomic writes) owns the champion. `run_layer4` is split into context-building, decision scoring and a pure gate function; serving uses the first two, training uses all three. Candidates live in `models/candidates/<id>/`; the existing FastAPI app gets a token-protected page and approve/reject routes.

**Tech Stack:** Python, pandas, LightGBM (existing), FastAPI + pytest (existing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-03-training-serving-design.md` (also read `2026-11-03-multi-store-design.md`, amended).

## Global Constraints

- Run tests from the repo root with `pytest` (`pytest.ini`: `pythonpath = src`, `testpaths = tests`).
- Serving never fits a model and never writes `models/registry` or `models/candidates`. Only `ModelRegistry.promote/rollback/import_legacy` write the registry.
- All registry/candidate file writes are atomic (temp file + `os.replace`); champion replacement never leaves a missing or truncated `champion_xlearner.pkl`.
- `XLearnerModel` / `TLearnerModel` stay defined in `src/layer4_uplift_margin.py` (existing pickles reference that module path).
- `store_id` regex `^[a-z0-9][a-z0-9-]{0,62}$`; a customer is the pair (store, customer_id); namespaced key is `f"{store_id}|{customer_id}"`.
- Gate target stays `CONFIG["gate_growth_target_pct"]` (3.0); holdout default 20%, seed 42; small-holdout warning threshold 100 customers.
- Approval token: `secrets.token_urlsafe(24)`, stored only as sha256 hex, compared with `hmac.compare_digest`. Approval routes are exempt from `X-API-Key`. Every value rendered into the page is HTML-escaped.
- Do not commit unless the user asks; "commit" steps are checkpoints (stage only the files named, never `git add -A`: the tree has unrelated untracked files). Work on a new branch from the current HEAD.
- Do not modify `train_*.py`, `retrain_plan.md`, the scheduler, Layers 1–3 logic, guardrail formulas.

## Review Focus

- Champion profit of 0 or negative in the gate (today a division by zero): must not crash; growth shown as `None`, advice fails safe.
- A store whose training split has no treated or no control rows: training stops with an error naming the store.
- Same `customer_id` in two stores: never merged; holdout membership is per pair.
- Store ids / approver names containing `<script>` or quotes: page renders them inert.
- Champion pickle changed on disk after approval (sha256 differs from manifest): serving refuses to run.

---

### Task 1: Atomic bytes write + ModelRegistry

**Files:** Modify `src/atomic_io.py`; Create `src/model_registry.py`, `tests/test_model_registry.py`.

**Interfaces:**
- Produces `atomic_write_bytes(path: str, data: bytes) -> None` (same temp+`os.replace`+mode logic as `atomic_write_text`).
- Produces in `model_registry.py`: `class RegistryError(Exception)`, `class StaleCandidate(RegistryError)`, `@dataclass(frozen=True) class ChampionRef: version: str; path: str; sha256: str; provenance: str`, and `class ModelRegistry(models_root: str)` with
  `current() -> ChampionRef | None`, `load_current() -> tuple[object, ChampionRef]`, `import_legacy() -> str`, `promote(model_bytes: bytes, expected_current: str | None, meta: dict) -> str`, `rollback(version: str) -> None`, `versions() -> list[dict]`.
- Layout: `<root>/registry/champion_xlearner.pkl`, `<root>/registry/manifest.json`, `<root>/registry/.lock`, `<root>/archive/<version>.pkl`. Manifest: `{"current": "v2", "versions": [{"version","sha256","created_at","approved_by","approved_at","provenance","metrics","legacy"}]}`; versions are `v1, v2, …`.

- [ ] **Step 1: Write failing tests** in `tests/test_model_registry.py` (use `tmp_path` as root; models are small picklable objects such as `{"n": 1}`):
  `test_import_legacy_makes_v1_with_provenance_simulated` (pickle exists, no manifest → `import_legacy()=="v1"`, `current().provenance=="simulated"`, manifest `legacy` true);
  `test_load_current_without_manifest_but_with_pickle_says_run_import_legacy` (RegistryError, message contains `--import-legacy`);
  `test_load_current_with_nothing_raises_no_champion` (message contains `train_champion.py`);
  `test_promote_archives_previous_and_replaces_atomically` (v1 imported, `promote(b, "v1", meta)=="v2"`, `archive/v1.pkl` equals old bytes, current is v2);
  `test_promote_with_stale_expected_current_raises_and_changes_nothing` (hash of registry dir before == after);
  `test_second_promote_against_same_base_is_refused` (two promotes with `expected_current="v1"`: second raises `StaleCandidate`);
  `test_first_champion_promote_with_expected_none`;
  `test_rollback_restores_old_bytes_and_manifest_current`; `test_rollback_unknown_version_raises`;
  `test_load_current_detects_modified_pickle` (append a byte → RegistryError mentioning `sha256`);
  `test_failed_replace_keeps_old_champion` (monkeypatch `os.replace` to raise during `promote`; champion bytes and manifest unchanged).
- [ ] **Step 2:** `pytest tests/test_model_registry.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement. `promote`/`rollback`/`import_legacy` take an exclusive `fcntl.flock` on `.lock`; order is archive-copy → atomic pickle write → manifest write last. `load_current` verifies the file's sha256 equals the manifest entry for `current`.
- [ ] **Step 4:** run the test file → PASS. Also `pytest tests/test_atomic_io.py` still passes.
- [ ] **Step 5:** Checkpoint commit of the named files.

### Task 2: Split `run_layer4` into context / decisions / gate (behavior-preserving)

**Files:** Modify `src/layer4_uplift_margin.py`; Create `tests/fixtures_pos.py`, `tests/test_layer4_refactor.py`.

**Interfaces:**
- Produces in `layer4_uplift_margin.py`:
  `@dataclass class Layer4Context: raw_df; cust_agg; features: list[str]; rfm_df; context_df; top_bundle_name: str; product_cogs: dict; default_cogs: float; label_mode: str  # "real"|"proxy"`;
  `build_layer4_context(pos_file_path: str, promo_response_path: str | None = None, product_cogs_dict: dict | None = None, default_cogs: float | None = None) -> Layer4Context` (everything in today's `run_layer4` from the file load down to `treat_data`/`ctrl_data`, minus fitting);
  `decide_for_customers(model, ctx: Layer4Context, customer_ids: set[str] | None = None) -> tuple[float, pd.DataFrame]` (today's `evaluate_model_profit`, plus the `should_send` column computed as today; `customer_ids` filters `cust_agg` before scoring);
  `market_is_recession(raw_df: pd.DataFrame) -> tuple[bool, float]` (today's baseline/current avg-spend block; returns `(is_recession, current_avg_spent)`);
  `@dataclass class GateResult: passed: bool; reason: str; growth_pct: float | None`;
  `gate_decision(challenger_profit: float, champion_profit: float | None, do_nothing_pct: float, is_recession: bool, current_avg_spent: float) -> GateResult` (today's branches verbatim, Thai reason strings unchanged; `champion_profit is None` → passed True, reason "บันทึกโมเดลนี้เป็น Champion ตัวแรก").
- `_train_xlearner` sets `model.feature_names_ = list(features)` on the returned model (both learner types).
- `tests/fixtures_pos.py`: `make_pos_csv(tmp_path, n_customers=120, seed=0, with_labels=False, name="pos.csv") -> str` producing a CSV that `_normalize_to_pos_format` accepts (read it first); `with_labels=True` adds `is_treated`, `post_promo_visit`, `promo_type`.

- [ ] **Step 1: Characterization test first** (before touching the module): `test_run_layer4_output_is_stable` runs today's `run_layer4` on `make_pos_csv(...)` with `models_root`-less call inside a `tmp_path` cwd with `monkeypatch.setattr("layer4_uplift_margin.REGISTRY_PATH", tmp_path/"models"/"registry")` and `monkeypatch.setattr("business_config.REGISTRY_PATH", ...)`; save the returned frame (`customer_id, recommended_action, expected_profit, should_send`) to `tests/golden/layer4_basic.csv`. Run it, confirm it passes and a second run gives identical output (if not, fix seeds in the fixture, not in the code).
- [ ] **Step 2:** Write failing tests: `test_decide_for_customers_matches_golden` (train via `_train_xlearner` on `build_layer4_context(...)` treat/ctrl slices, call `decide_for_customers`, compare to golden on the four columns); `test_decide_for_customers_filters_ids` (`customer_ids={"c001"}` → one row); `test_trained_model_has_feature_names` (`model.feature_names_ == ctx.features`); gate cases for `gate_decision`: growth +5% → passed; +1% → failed ("ยังไม่ถึงเกณฑ์"); −2% → failed; |Δ|<0.5% → failed; recession with `do_nothing_pct=60` → passed; `champion_profit=None` → passed; `champion_profit=0` and challenger 10 → no exception, `growth_pct is None`, `passed` False; `champion_profit=-50`, challenger −10 → no exception.
- [ ] **Step 3:** Extract the functions above; `run_layer4` temporarily calls them and keeps its current behavior (training + gate still in place) so the golden test stays green. Zero-champion-profit handling is the only intended behavior change.
- [ ] **Step 4:** `pytest tests/test_layer4_refactor.py tests/test_layer4_labels.py tests/test_layer4_customer_ids.py -v` → PASS.
- [ ] **Step 5:** Checkpoint commit.

### Task 3: Serving mode in `run_layer4`

**Files:** Modify `src/layer4_uplift_margin.py`, `src/csv_pipeline_engine.py` (main block around lines 485–560 only); Create `tests/test_layer4_serving.py`.

**Interfaces:**
- Consumes: `ModelRegistry` (Task 1), `build_layer4_context`, `decide_for_customers` (Task 2).
- Produces: `run_layer4(pos_file_path, promo_response_path=None, product_cogs_dict=None, default_cogs=None, models_root: str | None = None) -> pd.DataFrame` (same return columns as today). `models_root` defaults to `os.path.dirname(REGISTRY_PATH)`. It: builds the context; `ModelRegistry.load_current()`; verifies `champion.feature_names_ == ctx.features` (exact list and order) — a champion without `feature_names_` (legacy v1) is accepted on feature **count** with a `log("L4","legacy_champion_feature_names_unknown", level="WARN")`; scores; writes `data/processed/final_campaign_engine_output.csv` as today and `data/processed/model_used.json` (`{"champion_version","sha256","provenance"}`, atomic); logs `l4_serving_complete` with the version. It contains no `_train_xlearner` call, no gate, no legacy `champion_treat/ctrl` loading, no registry writes.
- `RegistryError`/feature mismatch propagate as exceptions; `csv_pipeline_engine.py` main catches `RegistryError`, prints the message and `sys.exit(1)`. Check the main block's write order so nothing derived from Layer 4 is written before the call.

- [ ] **Step 1: Failing tests** (`tmp_path` cwd and `models_root=tmp_path/"models"`, champion made by training on the fixture and `ModelRegistry.promote(..., expected_current=None, ...)`):
  `test_serving_leaves_models_dir_byte_identical` (hash tree before/after);
  `test_serving_never_calls_fit` (monkeypatch `layer4_uplift_margin._train_xlearner` and `lightgbm.LGBMClassifier.fit` to raise);
  `test_serving_output_equals_golden_when_champion_is_that_model`;
  `test_serving_writes_model_used_json_with_version`;
  `test_no_champion_raises_registryerror_and_writes_nothing` (no files under `data/processed`);
  `test_feature_name_mismatch_raises_naming_both_lists`; `test_same_count_different_order_raises`;
  `test_legacy_champion_without_feature_names_is_accepted_on_count`;
  `test_modified_champion_file_refused`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement as above; delete the old training/gate/champion-write code from `run_layer4` (training now lives in Task 5). **Step 4:** `pytest tests/test_layer4_serving.py tests/test_layer4_refactor.py -v` (the golden characterization test from Task 2 must be rewritten to call serving with a registered champion; update it in this step) → PASS. **Step 5:** checkpoint commit.

### Task 4: Pooled training data and holdout split

**Files:** Create `src/training_data.py`, `tests/test_training_data.py`.

**Interfaces:**
- Consumes `build_layer4_context`, `Layer4Context` (Task 2).
- Produces: `class TrainingDataError(Exception)`; `@dataclass class StoreInput: store_id: str; csv_path: str; labels_path: str | None`; `parse_store_data(arg: str) -> StoreInput` (`"<store_id>:<csv>[:<labels>]"`; invalid store id → `TrainingDataError`); `split_customers(customer_ids: Iterable[str], holdout_frac: float, seed: int) -> tuple[set[str], set[str]]` (deterministic: sort, then seeded permutation); `@dataclass class PooledData: contexts: dict[str, Layer4Context]; train_ids: dict[str, set[str]]; holdout_ids: dict[str, set[str]]; features: list[str]; label_mode: str`; `build_pooled(inputs: list[StoreInput], holdout_frac: float = 0.2, seed: int = 42) -> PooledData`; `training_frames(pooled: PooledData) -> tuple[pd.DataFrame, pd.DataFrame]` (treated rows, control rows from **training customers only**, with a `customer_key` column `store_id|customer_id`).

- [ ] **Step 1: Failing tests:** `test_parse_store_data_with_and_without_labels`; `test_parse_rejects_bad_store_id` (`..`, `A/b`, 64+ chars); `test_split_is_deterministic_and_disjoint_and_sized` (same seed → same sets; union = all; intersection empty; holdout size = round(frac·n)); `test_two_stores_with_same_customer_ids_are_not_merged` (both stores use `c000..`; `training_frames` keys unique, counts per store preserved); `test_no_holdout_customer_in_training_frames` (acceptance 4); `test_mixed_label_modes_across_stores_raise_naming_stores`; `test_store_without_treated_rows_in_training_raises_naming_store`; `test_empty_inputs_raise`.
- [ ] **Step 2:** FAIL. **Step 3:** implement; `label_mode` = common mode of contexts. **Step 4:** PASS. **Step 5:** checkpoint commit.

### Task 5: Candidate store, evaluation and report

**Files:** Create `src/candidates.py`, `tests/test_candidates.py`.

**Interfaces:**
- Consumes: Tasks 1, 2, 4.
- Produces in `candidates.py`: `evaluate_candidate(pooled: PooledData, challenger, champion, champion_ref: ChampionRef | None, training_meta: dict) -> dict` returning the **report** with exactly these keys: `candidate_id` (set by `create_candidate`), `created_at`, `compared_against_version` (`str|None`), `label_mode`, `provenance` (`"real-labels"|"proxy-labels"`), `holdout_size`, `holdout_warning` (bool, `< 100`), `stores` (list of `{store_id, rows, customers, holdout_customers}`), `champion` (`None` or metrics), `challenger` (metrics), `per_store` (`{store_id: {"champion": metrics|None, "challenger": metrics}}`), `gate` (`{passed, reason, growth_pct}`), `training` (`training_meta`: file names, row counts, date range, seed, holdout_frac). `metrics` = `{profit, send_count, blocked_count, do_nothing_pct}`. Both models are scored with `decide_for_customers(model, ctx, holdout_ids[store])` per store with default COGS (stated on the page), then summed.
  `create_candidate(models_root: str, challenger, report: dict) -> tuple[str, str]` → `(candidate_id, token)`; id format `YYYYMMDDTHHMMSSZ-<8 hex>`; writes `model.pkl`, `report.json`, `status.json` (`{"status":"pending","token_sha256":…,"approver":null,"decided_at":null}`).
  `load_candidate(models_root, candidate_id) -> dict` (`{"report","status"}`; `KeyError` if absent or id malformed); `check_token(models_root, candidate_id, token) -> bool`; `candidate_model_bytes(models_root, candidate_id) -> bytes`; `mark_candidate(models_root, candidate_id, status: str, approver: str) -> None` (`approved|rejected`, only from `pending`, else `RegistryError`).

- [ ] **Step 1: Failing tests:** `test_report_numbers_equal_recomputation_on_holdout` (acceptance 5: recompute with `decide_for_customers` and compare every metric); `test_report_has_all_keys_and_label_mode`; `test_first_champion_report_has_none_champion_and_gate_passes`; `test_holdout_warning_below_100`; `test_create_candidate_stores_only_token_hash` (token string not found in any file); `test_check_token_wrong_and_empty`; `test_mark_candidate_twice_refused`; `test_malformed_candidate_id_cannot_escape_dir` (`"../x"` → `KeyError`).
- [ ] **Step 2:** FAIL. **Step 3:** implement (gate via `market_is_recession` on the concatenated holdout `raw_df`s and `gate_decision`; `do_nothing_pct` from the challenger's pooled holdout decisions). **Step 4:** PASS. **Step 5:** checkpoint commit.

### Task 6: `train_champion.py` command (train, import-legacy, rollback)

**Files:** Create `src/train_champion.py`, `tests/test_train_champion_cli.py`.

**Interfaces:**
- Consumes Tasks 1, 4, 5 and `_train_xlearner`.
- Produces `main(argv: list[str] | None = None) -> int`. Modes: default train (`--store-data` repeatable, `--holdout-frac`, `--seed`, `--models-root` default `os.path.dirname(REGISTRY_PATH)`), `--import-legacy`, `--rollback <version>`. Train mode: `build_pooled` → `_train_xlearner(*training_frames(...), pooled.features)` → load current champion if any → `evaluate_candidate` → `create_candidate` → log `train_candidate_created` → print the link `f"{CRM_API_PUBLIC_URL or 'http://localhost:8000'}/v1/models/candidates/{id}?token={token}"` plus a line saying the API must be running to open it. It never modifies `registry/`. Errors (`TrainingDataError`, `RegistryError`) print the message and return 1.

- [ ] **Step 1: Failing tests** (call `main([...])` with `capsys`): `test_train_creates_candidate_and_prints_link_without_touching_registry` (hash registry before/after; one candidate dir; link contains the id and token; `CRM_API_PUBLIC_URL` honored via monkeypatch); `test_train_with_two_stores_lists_both_in_report`; `test_train_with_no_champion_marks_first_champion`; `test_import_legacy_then_train_compares_against_v1`; `test_rollback_cli_restores_version` and `test_rollback_unknown_version_returns_1`; `test_bad_input_returns_1_and_creates_no_candidate`; `test_research_scripts_do_not_write_champion_xlearner_pkl` (read each `src/train_*.py` source; assert none contains the literal `champion_xlearner.pkl"` as a write target — `train_rlearner.py` only reads it; assert on `open(..., "wb")` paths).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** checkpoint commit.

### Task 7: Approval page and routes in the CRM API

**Files:** Modify `src/crm_api/settings.py` (add `models_dir: Path = Path("models")`, `public_url: str = "http://localhost:8000"`; env `CRM_API_MODELS_DIR`, `CRM_API_PUBLIC_URL`; keep existing constructor calls valid), `src/crm_api/app.py`; Create `src/crm_api/models.py`, `src/crm_api/model_page.py`, `tests/crm_api/test_models_approval.py`.

**Interfaces:**
- Consumes Tasks 1 and 5; `ApiError` from `crm_api/errors.py`.
- Produces: `model_page.render_candidate_page(candidate_id: str, report: dict, status: dict, current_version: str | None, token: str) -> str` (pure; page fields exactly as in the spec "Comparison page": candidate id, training data, label mode, holdout size and warning, provenance, two columns Champion/Challenger with profit, % change, send count, guardrail-blocked, per-store counts, gate advice, proxy-labels caution block, shared-aggregates note, "evaluated with default COGS" note, approver-name input and the two buttons calling `fetch` POST with JSON; when status is not pending the buttons are replaced by the decision; when `report.compared_against_version != current_version` a stale banner replaces the approve button).
  Routes (router without `require_key`): `GET /v1/models/candidates/{candidate_id}?token=…` → HTML; `POST /v1/models/candidates/{candidate_id}/approve` and `/reject`, body `{"token": str, "approver": str}` (approver required, 1–80 chars). Errors in the existing `{"error": {code,message}}` shape: malformed/unknown id → 404 `candidate_not_found`; bad/missing token → 401 `unauthorized`; not pending → 409 `candidate_not_pending`; stale → 409 `candidate_stale`. Approve = `ModelRegistry.promote(model_bytes, report["compared_against_version"], meta)` then `mark_candidate`, then `log("L4","champion_approved", …)`; returns `{"status":"approved","version":"vN"}`.

- [ ] **Step 1: Failing tests** (`TestClient(create_app(Settings(... models_dir=tmp_path)))`, candidates made through Task 5/6 helpers): `test_page_requires_valid_token` (401 wrong/missing; 404 unknown/`..`); `test_page_shows_all_fields` (assert id, both profits, label mode, holdout size, store ids, gate reason present); `test_proxy_mode_page_shows_caution_text`; `test_page_escapes_hostile_store_id_and_approver` (`<script>alert(1)</script>` appears only as `&lt;script&gt;`); `test_approve_promotes_archives_logs_and_next_serving_sees_new_version` (acceptance 7: `archive/v1.pkl` exists, manifest current `v2`, `output/pipeline_audit.jsonl` line, `load_current()` version `v2`); `test_approve_wrong_token_changes_nothing`; `test_approve_twice_second_is_409_not_pending`; `test_two_candidates_same_base_second_approval_is_409_stale`; `test_rollback_then_old_candidate_is_stale`; `test_reject_keeps_champion`; `test_approve_requires_approver_name` (422); `test_other_routes_still_require_api_key`.
- [ ] **Step 2:** FAIL. **Step 3:** implement; redirect nothing, no template engine (f-strings with `html.escape`). **Step 4:** `pytest tests/crm_api -v` → all PASS (existing tests unchanged). **Step 5:** checkpoint commit.

### Task 8: Docs and end-to-end check

**Files:** Create `docs/training.md`; Modify `docs/crm-api.md` (model routes + new env vars); Create `tests/test_training_serving_e2e.py`.

- [ ] **Step 1: Failing test** `test_full_flow`: two fixture shops → `train_champion.main` (no champion yet) → `TestClient` approve → serving run with `run_layer4(models_root=...)` for shop A writes `model_used.json` with `v1`… then a second `train_champion` run → candidate compared against `v1`, approve → serving shows `v2` → `--rollback v1` → serving shows `v1`; assert `models/` tree unchanged across each serving run.
- [ ] **Step 2:** FAIL, **Step 3:** fix any wiring gaps found; write `docs/training.md` (commands: first-time `--import-legacy`, train with `--store-data a:csv:labels --store-data b:csv`, open link, approve, rollback, serve; caveats: proxy labels, default COGS in comparison, token-in-URL, localhost bind, pooling consent and "cannot unlearn").
- [ ] **Step 4:** `pytest` (whole suite) → PASS; run `uvicorn` per `docs/crm-api.md` with `CRM_API_KEY` set, open a real candidate link in a browser and press Approve once on a throwaway `--models-root`. **Step 5:** checkpoint commit.

---

## Self-review

- **Spec coverage:** goals 1–5 → Tasks 3, 6, 7, 1+6, 5+7; acceptance 1 → T3; 2 → T3; 3 → T6; 4 → T4; 5 → T5; 6 → T7; 7 → T7; 8 → T1; 9 → T1+T6; 10 → T2/T3 (updates golden) + whole-suite run in T8; 11 → T8. Calls 1–9: import v1 (T1/T6), version-id stale (T1/T7), shared aggregates (note on page, T7), feature names (T2/T3), callers (T3/T6 test), token (T5/T7), proxy allowed (T7), `--labels` route (T4), no multi-store doc change (none).
- **Deviations to tell the user:** comparison profits use default COGS (CONFIG), not the per-product COGS the monthly plan uses; legacy v1 is checked by feature count only until the first new champion; pooled training stops if shops mix real and proxy labels.
- **Type consistency:** `ChampionRef`, `Layer4Context`, `PooledData`, report keys and route/error codes are defined once above and used by name in later tasks.
- **Proportion:** the plan lists signatures, test names and values; bodies are left to the implementer.

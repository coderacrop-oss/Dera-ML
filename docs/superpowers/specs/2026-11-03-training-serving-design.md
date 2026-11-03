# Training / Serving split with human-approved champion — design

Date: 2026-11-03 · Status: draft for review. Builds on `2026-11-03-crm-api-design.md` (the API that exists) and
`2026-11-03-multi-store-design.md` (draft; defines `--score-only` and the shared frozen champion — see "Relationship to other designs").
Items marked **[CALL]** are my choices, not agreed with the user. Everything else was stated or approved by the user.

## Why
Today every run of `csv_pipeline_engine.py` trains a new X-learner inside Layer 4 (`run_layer4`, `layer4_uplift_margin.py`),
scores the customers **with that new model (the "Challenger")**, and, if it passes a gate, overwrites the stored champion
(`models/registry/champion_xlearner.pkl`). The stored champion is only loaded to compare profit; it never scores anyone.
So: (a) every batch from the CRM retrains, (b) a model replaces the champion with no human looking, and (c) the comparison
is done on the same customers the model was trained on (no holdout), so the challenger looks better than it is.
The user wants the way large ML systems work: **training is a separate, occasional job; serving only uses the approved champion;
a person approves a new champion after seeing a comparison.**

## Goals
1. **Serving is score-only.** A pipeline run loads the approved champion and scores the batch. It never trains and never writes
   to the model registry. (Layers 0–3 and 5, the scheduler and the API are unchanged; the work stays batch, not real-time.)
2. **Training is its own command**, run by hand for now (target cadence: monthly once the system is finished; automation is later).
   It trains a Challenger, evaluates Challenger and current Champion **on customers held out from training**, and produces a
   comparison report. It never changes the champion.
3. **A person approves.** The training command ends by printing a link to a comparison page (layout "A" below). The champion changes
   only when someone presses *Approve* on that page.
4. **Rollback.** Every champion that was ever approved stays available by version; the previous one can be restored.
5. **Honest numbers.** The comparison page says which kind of labels the evaluation used and that profits are model estimates.

## Non-goals
- Real-time (one customer per call) scoring. Batch only.
- Automatic scheduled training, drift monitoring, automatic promotion. (Cadence is manual now.)
- A/B or shadow rollout of a new champion in production.
- Per-shop models. One shared champion, trained on all shops' pooled data (user's decision), serves every shop.
- Notifications (email/LINE). The link is printed in the terminal only.
- Changing Layer 1–3 logic, the guardrail formulas, or the scheduler.
- User accounts / login. Approver identity is a free-text name (see Approval).

## Anti-goals (a failure even if it "works")
- Any code path where a serving run (or the API) changes `models/registry`.
- Approving a candidate that was compared against a champion that is no longer the current one (stale approval).
- A comparison page whose numbers differ from what the training job computed, or that hides that the evaluation used proxy labels.
- A half-written champion file visible to a concurrent serving run.

## Relationship to other designs
- `multi-store-design` (draft, not implemented: no `--score-only` exists in the code today) already specifies `--score-only` and a
  shared frozen champion, and leaves "how and when a real model replaces the simulated one" as a later decision.
  **This design is that later decision.** Conflict to resolve while planning: multi-store keeps "training mode = default, unchanged".
  Here, **no mode of the engine may overwrite the champion**, so the engine's old train-and-gate-in-the-run behavior is removed and
  scoring against the champion becomes the engine's behavior (multi-store's `--score-only` flag then becomes redundant or an alias).
  [CALL] This design wins on that point because the user explicitly asked that only a person promotes a champion. The user should be
  told when the plan is written, since it changes what multi-store's acceptance criterion 4 means.
- `crm-api-design`: API stays read-only **for shop data**. The approval page adds two write actions on model candidates only.
  Per the multi-store design, models are shared across shops, so these actions are operator-level, not shop-scoped.

## Design

### Units
1. **Training command** [CALL] `python src/train_champion.py --csv <transactions.csv> [--labels <promo_labels.csv>] [--holdout-frac 0.2] [--seed 42]`.
   The operator names the training data explicitly; nothing is gathered automatically from shops. It reuses the feature/label
   construction and `_train_xlearner` from `layer4_uplift_margin.py` (extract them so training and serving share one definition of
   features; builder decides how).
2. **Holdout evaluation.** Customers are split **by customer id** [CALL] (seeded, default 20%), so no holdout customer's rows are in
   training. Both models score the holdout customers with the same decision/guardrail code used in serving
   (today `evaluate_model_profit`; extract it so serving and evaluation use identical logic). Reported per model:
   estimated profit, number of customers with `should_send=1`, number blocked by the CFO guardrail, "Do Nothing" share.
   The existing gate (`gate_growth_target_pct`, recession rule) is reused to produce a pass/fail **advice**; it no longer acts alone.
3. **Candidate store** [CALL] `models/candidates/<candidate_id>/` containing `model.pkl`, `report.json`, `status.json`
   (`pending|approved|rejected`), and a hash of the champion version it was compared against. `candidate_id` = UTC timestamp + short hash.
4. **Registry** [CALL] versioned: `models/registry/champion_xlearner.pkl` stays the file serving reads (current champion).
   `models/registry/manifest.json` lists versions (id, trained-at, approved-by, approved-at, data fingerprint, label mode, metrics).
   Superseded champions are copied to `models/archive/<version_id>.pkl` **before** replacement. All writes use `atomic_io`.
   The existing `champion_akeed_mock_full.pkl` in `archive/` is left as is.
5. **Serving run.** `csv_pipeline_engine.py` loads the champion once at start of Layer 4, checks its feature count against the CSV's
   features, scores, and writes the same outputs as today. No champion, or feature mismatch → exit non-zero with a message naming the
   cause and pointing to the training command; it never trains as a fallback. The run records the champion version id in its output
   (so a plan can be traced to a model) [CALL: in `campaign_summary_*.json` / pipeline audit log, builder picks the field].
6. **Approval page** (served by the existing FastAPI app) [CALL]:
   - `GET /v1/models/candidates/{candidate_id}?token=…` → the comparison page (HTML).
   - `POST /v1/models/candidates/{candidate_id}/approve` and `/reject` with the same token and an approver name field.
   - A browser opening a link cannot send the `X-API-Key` header, so, like the calendar feed, the link carries a **per-candidate
     random token** (stored hashed in `status.json`, printed once with the URL by the training command). Anyone with the link can
     act on that one candidate; there is no other login. Default bind stays localhost; exposing it to other machines is a later step.
   - *Approve* is refused with `409 candidate_stale` if the current champion is not the one the candidate was compared against, or if the
     candidate is no longer `pending`. Approve = archive current champion → atomically replace → update manifest → log to
     `output/pipeline_audit.jsonl`.
   - *Rollback* [CALL] is a command (`python src/train_champion.py --rollback <version_id>`), not a button, for now.
7. **Link printed when training ends**: `http://<host>:<port>/v1/models/candidates/<id>?token=<token>` where host/port come from env
   `CRM_API_PUBLIC_URL` (default `http://localhost:8000`). If the API is not running the link will not open; the command says so.

### Comparison page (user chose layout "A": one page, side by side)
Header: candidate id, training data (file name, row count, date range), label mode, holdout size, `model_provenance` (e.g. simulated).
Two columns, **Champion (current vN)** and **Challenger**: estimated profit (with % change), customers to send, customers blocked by
guardrail. Below: gate advice (pass/fail with the numbers), a caution block, and two buttons: *Approve → set as Champion* and
*Don't approve (keep current)*, plus an approver-name field. If there is no champion yet, the left column says so and the gate advice
says "first champion". Numbers come from `report.json` only; the page computes nothing.
The per-customer difference table ("layout B") was considered and **not chosen**; do not build it.

### Evaluation honesty (important constraint)
Without real promotion-response labels the engine builds proxy labels: treatment is **random** (seed 42, ~40%) and "converted" is
"came back in the last 25% of the period" (see `run_layer4`). Uplift evaluated on random treatment has no real causal signal
(the code's own comment says AUUC ≈ 0 is the honest answer). So the report **must** show the label mode ("real promotion labels" vs
"proxy labels") and, for proxy mode, a visible caution that the comparison is not evidence of real campaign benefit.
Approval is still allowed (the user decides), but the page must not present proxy-mode numbers as a real improvement.

## Acceptance criteria
1. A serving run (with a valid champion) leaves `models/registry/` and `models/candidates/` byte-identical (hash before/after) and calls
   no fitting code (test by making the fit function raise).
2. A serving run with no champion, or with a feature-count mismatch, exits non-zero, names the cause, writes no outputs, trains nothing.
3. The training command never changes `champion_xlearner.pkl` or `manifest.json`; it creates one candidate directory and prints the link.
4. No holdout customer id appears in the training rows (test on a fixture); the same seed gives the same split.
5. `report.json` numbers equal a recomputation of both models on the holdout using the serving decision code (test).
6. The page shows every field listed under "Comparison page", including label mode and provenance; proxy-mode shows the caution.
7. Approve with a wrong/missing token → 401/403 and no change. Approve of a non-pending or stale candidate → 409 and no change.
   Approve on a valid pending candidate: previous champion is in `archive/`, manifest updated, audit log line written, and the next
   serving run reports the new version id.
8. Killing the process mid-approve never leaves a missing or truncated `champion_xlearner.pkl` (atomic replace; test with a simulated failure).
9. Rollback to a listed version restores that file and updates the manifest; an unknown version id fails clearly.
10. Existing tests still pass, with tests that assumed the old train-in-run behavior updated; new tests cover items 1–9.
11. `docs/crm-api.md` (or a new `docs/training.md`) documents: how to train, how to open and approve, how to roll back, how to serve.

## Calls settled after the builder check (all [CALL]; the user may overturn them)
1. **Today's champion.** The existing `models/registry/champion_xlearner.pkl` is imported into the manifest as version `v1`,
   labelled `legacy-unverified` (no holdout report, provenance "simulated" per the multi-store design) and keeps serving until a
   candidate is approved. A one-time migration step creates the manifest; if the manifest is missing and the pickle exists, serving
   refuses to start and says to run the migration (no silent guess).
2. **Stale check** compares the champion **version id** the candidate was evaluated against with the current one. A rollback changes
   the current version, so candidates trained before a rollback become stale and must be re-trained.
3. **Holdout and shared aggregates.** The model is fitted only on training customers' rows. Layers 1–3 (RFM, thresholds, margins) are
   computed once over the whole CSV, exactly as in serving, where the batch is always scored with batch-wide aggregates. This is
   accepted leakage of *unsupervised* aggregates (no labels pass through them) and is written on the page's caution block. It is not
   the in-sample problem of today, where the fitted model is also judged on its own training customers.
4. **Feature compatibility.** The champion stores its feature **names and order** (not only the count); serving verifies the exact
   list and fails otherwise. Legacy T-learner files (`champion_treat.pkl` / `champion_ctrl.pkl`) are no longer read by serving.
5. **Callers.** In `src/`, only `csv_pipeline_engine.py` calls `run_layer4`; the existing tests exercise `_load_real_labels` and id
   normalisation, not training, so they are expected to keep passing. The training/gate section of `run_layer4` moves to the training
   command; the `train_*.py` research scripts keep their own differently named output files and must not write
   `champion_xlearner.pkl` (add a test or grep check). `retrain_plan.md` is not touched.
6. **Approval link security.** The token does not expire in this version (candidates are acted on once; a used or rejected candidate
   ignores it). The approval routes are exempt from the `X-API-Key` dependency and rely on the token (same pattern as the calendar
   feed). The token is sent in the URL for the page and in the POST body for the actions. Because it can appear in access logs and
   shell history, keep the default localhost bind and document this.
7. **First champion / proxy labels.** A first approval is allowed with proxy labels; the page shows the caution and the person
   decides. Holdout under the minimum size warns and does not block. Proxy mode is expected to be the common case for now.
8. **Normal training input.** The `--labels` file (the form `_load_real_labels` reads) is the route to meaningful numbers; without
   it the command uses the proxy labels from `run_layer4`.
9. The `CRM_API_PUBLIC_URL` default and the conflict with multi-store criterion 4 need no change to the multi-store doc; the conflict
   is noted here and raised when the plan is written.

## Decisions left to the builder
Module layout and names; how to extract shared feature/decision code; the exact `report.json` and `manifest.json` schemas; token
generation and hashing; HTML templating approach for the page (a single static-style page is enough); where the champion version id
is recorded in outputs; how a first-ever champion (no champion yet) is bootstrapped through the same approval flow.

## Open points and risks
- **Training data (decided by the user, 2026-11-03):** the ML team trains the champion from the data of **all shops pooled**; all
  shops record data in the same format. Consequences written into the design:
  - The training command takes several inputs, one per shop (CSV + optional labels file, each tagged with its `store_id`) [CALL:
    repeated `--store-data <store_id>:<csv>[:<labels>]`; exact CLI is the builder's]. Features are built per shop with the same code
    as serving, then concatenated.
  - A customer is the pair (store, customer_id) everywhere in training, the split and the report, so equal ids in two shops never
    collide. The holdout is split by that pair.
  - `report.json` and the page list **which shops were included, with row and customer counts**, so the data used is auditable.
  - Shops differ in size and price level; pooled training can be dominated by large shops. The page shows per-shop customer counts
    and the builder should report the holdout profit estimate per shop as well as in total (informational, not part of the gate).
  - Pooling is a step beyond the multi-store draft, which said shop data never trains the model; that draft is amended accordingly.
  - Removing a shop later stops its data from entering future trainings, but cannot erase what a model already learned from it.
    The user should know this when deciding which shops are included. Whether each shop's contract allows pooling is the user's
    responsibility; the system only records which shops each candidate used.
- Holdout size can be small for small shops; the page shows the holdout size and warns below a threshold [CALL: 100 customers].
- Estimated profit is a model estimate, not realized revenue (same caveat as the research paper).
- Pickle files are loaded from `models/`; this is safe only because only local operators write there. Do not add an upload path.
- Two trainings at once, or approving while a serving run starts: serving loads the champion once; replacement is atomic; stale check
  covers two candidates compared against the same champion (the second approval is refused).

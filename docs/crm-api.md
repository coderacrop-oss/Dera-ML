# CRM Campaign API

Read-only REST API over the files the Campaign Engine writes in `data/processed/`.
Design: `docs/superpowers/specs/2026-11-03-crm-api-design.md`.

## Run
```bash
pip install -r requirements-api.txt            # plus numpy/pandas (already needed by the pipeline)
export CRM_API_KEY=change-me                    # required; the server refuses to start without it
export CRM_API_CALENDAR_TOKEN=another-secret    # optional; read-only token for calendar.ics only
export CRM_API_DATA_DIR=data/processed          # optional (default shown)
export CRM_API_MODELS_DIR=models                # optional; where model candidates/registry live (default shown)
export CRM_API_PUBLIC_URL=http://localhost:8000 # optional; base of the approval link printed by train_champion.py
uvicorn crm_api.app:app_factory --factory --app-dir src --port 8000
```
Interactive docs: `http://localhost:8000/docs`.

The API also hosts the champion approval page (`/v1/models/candidates/…`, token-protected, hidden from `/docs`); see `docs/training.md`.

## Produce a month (pipeline → scheduler → API)
```bash
python src/csv_pipeline_engine.py <transactions.csv>   # writes broadcast_list.csv + segments_latest.json
python src/campaign_scheduler.py --month 2026-12       # writes campaign_schedule/summary/holdout + segments_2026-12.json
```
Re-running a month overwrites its files (atomic writes). The API re-reads files when their mtime/size
changes; clients can use the `ETag` header / `If-None-Match` (→ `304`) to detect changes.

## Endpoints (header `X-API-Key`, except `/health` and the token feed)
| Path | Returns |
|---|---|
| `GET /health` | status + months available (no auth) |
| `GET /v1/months` | months that have a plan |
| `GET /v1/plans/{month}` | totals, waves (+weeks), peak hours, gate status |
| `GET /v1/plans/{month}/events?from=&to=&wave=` | calendar events (date, wave, window, count, actions, profit) |
| `GET /v1/plans/{month}/calendar.ics` | iCalendar feed; also accepts `?token=$CRM_API_CALENDAR_TOKEN` |
| `GET /v1/plans/{month}/segments` | customer groups (needs `segments_{month}.json`, else `segments: null`) |
| `GET /v1/plans/{month}/promotions` | distinct promotions with counts and an example message |
| `GET /v1/plans/{month}/events/{event_id}/customers?limit=&cursor=` | recipients of one event (paged, keyset cursor) for batch sending; `409 plan_changed` if the plan is updated mid-read |
| `GET /v1/plans/{month}/customers/{id}` | that customer's scheduled offer; `404` for holdout/suppressed/unknown |

### Waves and batches
A *wave* is a group of customers by promotion type (1 off-peak traffic, 2 peak upsell, 3 VIP), **not** one send.
Each wave is spread over several days (a different batch of customers per day) and every customer is scheduled
exactly once per month. Each event carries `batch_index`, `batch_total` and `wave_customer_total` so a UI can show
"batch 3 of 10 · 22 people (209 in the wave)".

### Batch sending flow (the CRM does the sending)
1. `GET /v1/plans/{month}/events` → each event has `start`, `wave`, `customer_count` and a ready-made `customers_url`.
2. At/before `start`, `GET customers_url?limit=1000`, then follow `next_cursor` until it is `null`.
3. Send each recipient's `message` on your own channel. On `409 plan_changed`, restart that event from page 1.
The API does not send anything, has no "sent" state and does not receive outcomes.

Errors: `{"error": {"code", "message"}}` with `401/404/422/503`.
`503 data_unavailable` is retry-able: a file is missing/unparseable, or the files of a month disagree with the
row counts in `campaign_summary_*.json` (e.g. read while the pipeline is mid-update). The API never serves
a partial plan. Derived views (events, promotions, `.ics`, customer index) are computed once per loaded
month and refreshed automatically when the files change.

```bash
curl -H "X-API-Key: $CRM_API_KEY" localhost:8000/v1/plans/2026-12
curl "localhost:8000/v1/plans/2026-12/calendar.ics?token=$CRM_API_CALENDAR_TOKEN"
```

## Tests
`pytest` from the repo root. `tests/crm_api/test_contract_vs_pipeline_output.py` compares the API with
the real files in `data/processed/` and is skipped when none exist.

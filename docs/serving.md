# Serving

## Design

```text
ingest worker ──▶ store/snapshots/<id[:2]>/<id>.json.gz   one prebuilt index per patient
                                 │  (shared storage: a volume now, object storage later)
                                 ▼
load balancer ──▶ API replica ×N: IndexCache (LRU) ──▶ catalog / search / timeline
                                    │
                                    └─▶ /ask: agent loop ──▶ OpenAI-compatible LLM server
```

- **Prebuilt snapshots.** Ingest writes each patient's built `PatientIndex` as gzip-compressed JSON (`snapshot.py`), after the batch's Parquet files and manifest are committed. Serving fetches one object by patient id instead of querying tables, then rebuilds the in-memory search structures.
- **Replicas hold no state of their own.** An API replica holds only a bounded LRU cache. A cached entry is reused while its snapshot file is unchanged and reloaded after a re-ingest, so any number of replicas can sit behind a load balancer. Routing a patient to the same replica raises cache hits but is not needed for correctness.
- **Retrieval stays in-process.** `/ask` runs the agent loop in the API process and calls the tools as local functions, so the several tool calls in one answer add no network hops. Only the LLM is remote.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/readyz` | snapshot directory present, cache stats for this worker |
| GET | `/v1/patients/{id}/catalog` | catalog text, snapshot metadata, `catalog_tokens` once counted |
| POST | `/v1/patients/{id}/search` | `{query, k, kinds}`: tool text plus structured hits |
| GET | `/v1/patients/{id}/timeline/{concept_id}?since=&until=` | timeline text with resource ids |
| POST | `/v1/patients/{id}/ask` | `{question, max_steps}`: answer, tool steps, model, seconds |

Status codes: an unknown patient or concept returns 404, an invalid `kinds` returns 422, and an unreachable or failing LLM server returns 502.

```bash
uv run python -m health_context ingest data --store store
uv run --group serve python -m health_context serve --store store --workers 4
curl -s localhost:8000/v1/patients/f5749532-3295-ad01-d9b5-932c997e7a01/catalog
```

Configuration comes from environment variables: `SNAPSHOT_DIR`, `INDEX_CACHE_SIZE` (default 256), and `LLM_BASE_URL` / `LLM_MODEL` for `/ask`.

## Measured on this machine (12 cores, local SSD, client on the same host)

| | result |
|---|---|
| snapshot size, 109 patients | 32 KB median, 766 KB max, 5.3 MB total |
| snapshot load (decompress and rebuild) | 4.3 ms median, 52 ms max (Parquet path: 13 ms / 114 ms) |
| snapshot output vs bundle output | identical for all 109 patients |
| ingest cost of writing snapshots | about 20% (1.02 s to 1.22 s for the corpus) |
| 4 workers, mixed catalog/search, 109 patients, concurrency 32 | 848 req/s, p50 35 ms, p95 61 ms, p99 100 ms |
| first requests while caches are empty, concurrency 16 | p95 116 ms |
| `/ask` with the LLM server down | 502 in 15 ms |

These numbers exclude the LLM. With a model in the loop, each answer takes several sequential model calls, and model serving sets the end-to-end latency.

## Limits and next steps

- **Object storage.** `SnapshotStore` reads a filesystem path. An S3 or GCS backend needs the same `get`/`put`/`path` methods, using the object's ETag where the cache now uses the file's mtime.
- **Token counts.** `catalog_tokens` stays empty until ingest counts catalogs with Anthropic `count_tokens`, which is blocked by the account's credit balance.
- **Concurrent re-ingest.** Two workers re-ingesting the same patient at once race on its snapshot, and the last writer wins. A queue keyed by patient id avoids this.
- **Access control is not implemented.** Real records need authentication, per-patient authorization, and audit logging of every tool call and resource id served.

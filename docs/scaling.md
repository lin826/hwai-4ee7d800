# Scaling to Terabytes

## Target

About 2 TB more FHIR data. At this corpus's mean of 3.4 MB per bundle that is roughly 600k patients and on the order of 700M resources: about 5,500 times the 109-patient sample. That does not fit in memory or on one laptop disk, but bundles are independent, so the work is split per patient.

## Design

```text
bundles (local disk or object storage, unchanged = source of truth)
   │  ingest: process pool, one batch of bundles per task, no coordination
   ▼
PatientIndex per bundle (same code as in-memory retrieval)
   │  flatten to rows
   ▼
store/{events,notes,resources,manifest}/batch-<id>.parquet   (zstd, rows grouped by patient)
   │
   ├─ serve:  manifest → patient's batch → read 3 files filtered by patient_id → PatientIndex → tools
   └─ analyze: SQL over all files (views keep only each patient's current batch)
```

- **One code path.** Ingest flattens the same `PatientIndex` that in-memory retrieval uses, and serving rebuilds it from rows. A test and a full-corpus check confirm the catalog, every timeline and search output are identical either way.
- **Append-only, incremental.** A bundle with an unchanged path, size and mtime is skipped. A changed bundle is written into a new batch, and the newest manifest row wins. Old files are never rewritten, so a failed worker cannot corrupt existing data. The manifest is written last, so a batch becomes visible only when all its data files exist.
- **Exact values kept.** `value_json` round-trips the original value. `value_num` is there for SQL analytics.
- **Portable files.** Parquet is readable by DuckDB, Spark, Athena, BigQuery and Snowflake, so changing engines later does not mean re-ingesting.

## Measured on this machine (12 cores, local SSD)

| | result |
|---|---|
| ingest, 109 bundles / 366 MB | 1.0 s, 359 MB/s |
| store size | 4.6 MB (about 80x smaller than the JSON) |
| load one patient from store | 13 ms median, 114 ms for the largest bundle |
| load one patient, 600k-patient manifest in 1,000 files | about 40 ms |
| retrieval eval from store vs from JSON | identical (1,577 cases) |

## Extrapolation to 2 TB (estimates, not measured)

- **Ingest time.** Linear in bytes, because there is no cross-patient work: about 1.6 hours on this 12-core laptop, or minutes on a cluster sharded by file list.
- **Store size.** About 25 GB at this compression ratio. Real records with less templated notes will compress less, so plan for 25 to 100 GB.
- **Serving latency.** Latency depends on one patient's size, not on corpus size, because a lookup reads one batch's files filtered by patient id.

## Known limits and next steps

1. **Compaction.** Superseded batches stay on disk (hidden from views). Add a `compact` job that rewrites live rows into fewer, larger files sorted by patient id.
2. **Batch sizing.** At 2 TB, use `--batch-mb` around 1 to 2 GB so there are about 1,000 to 2,000 batches, not tens of thousands of small files.
3. **Object storage.** Listing many manifest files per request is slow on S3. Cache the current patient-to-batch map in memory or a key-value store when the service starts.
4. **Very large single bundles.** Each bundle is parsed with `json.load`. Bundles of several GB would need a streaming parser such as `ijson`.
5. **Orchestration.** The ingest is a pure function from file batch to Parquet files, so it maps directly onto a Dagster partitioned asset (one partition per batch) when scheduling, retries and lineage are needed.
6. **Not yet ingested:** shared practitioner and organization files, billing and supply resources.

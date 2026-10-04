"""Columnar store for many patients: parallel ingest to Parquet, DuckDB reads.

Layout (one file per table per ingest batch, rows grouped by patient)::

    store/
      manifest/batch-<id>.parquet   patient -> batch, source file, Patient JSON
      events/batch-<id>.parquet     one row per concept event (a lab reading, a diagnosis, ...)
      notes/batch-<id>.parquet      one row per de-duplicated note section occurrence
      resources/batch-<id>.parquet  every resource id and type, for citation
      snapshots/<id[:2]>/<id>.json.gz  prebuilt per-patient index for serving

Bundles are independent, so ingest is a map over files: each worker turns a
batch of bundles into rows with the same ``PatientIndex`` used in memory and
writes its own files, with no coordination. The manifest is append-only; the
newest manifest row for a patient names the batch holding its current rows,
so re-ingesting a changed bundle never rewrites old files. Serving one patient
reads only that batch's files, filtered by patient id; corpus-wide analytics
are plain SQL over all files.

Parquet paths can be local directories or object storage URLs that DuckDB
reads (``s3://...`` with the httpfs extension); only local paths are tested.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .index import Concept, Event, NoteChunk, PatientIndex
from .pipeline import load_bundle
from .snapshot import SnapshotStore

SCHEMA_VERSION = 1
TABLES = ("manifest", "events", "notes", "resources")

SCHEMAS = {
    "manifest": pa.schema(
        [
            ("patient_id", pa.string()),
            ("bundle_id", pa.string()),
            ("batch_id", pa.string()),
            ("source_path", pa.string()),
            ("source_size", pa.int64()),
            ("source_mtime_ns", pa.int64()),
            ("patient_json", pa.string()),
            ("ingested_at", pa.timestamp("us", tz="UTC")),
            ("schema_version", pa.int32()),
        ]
    ),
    "events": pa.schema(
        [
            ("patient_id", pa.string()),
            ("concept_id", pa.string()),
            ("kind", pa.string()),
            ("display", pa.string()),
            ("system", pa.string()),
            ("code", pa.string()),
            ("category", pa.string()),
            ("date", pa.string()),
            ("resource_id", pa.string()),
            ("resource_type", pa.string()),
            # value_num serves SQL analytics; value_json round-trips the exact
            # original value (int vs float, coded text, booleans).
            ("value_num", pa.float64()),
            ("value_json", pa.string()),
            ("unit", pa.string()),
            ("status", pa.string()),
            ("end", pa.string()),
            ("encounter_id", pa.string()),
            ("detail", pa.string()),
        ]
    ),
    "notes": pa.schema(
        [
            ("patient_id", pa.string()),
            ("chunk_id", pa.string()),
            ("section", pa.string()),
            ("text", pa.string()),
            ("ordinal", pa.int32()),
            ("visit", pa.string()),
            ("date", pa.string()),
            ("resource_ids", pa.list_(pa.string())),
        ]
    ),
    "resources": pa.schema(
        [
            ("patient_id", pa.string()),
            ("resource_id", pa.string()),
            ("resource_type", pa.string()),
        ]
    ),
}


# -- ingest -----------------------------------------------------------------


def index_rows(
    index: PatientIndex, source: Path, batch_id: str
) -> dict[str, list[dict]]:
    """Flatten one patient's index into rows for each table."""
    pid = index.patient_id
    stat = source.stat()
    rows: dict[str, list[dict]] = {t: [] for t in TABLES}
    rows["manifest"].append(
        {
            "patient_id": pid,
            "bundle_id": index.bundle_id,
            "batch_id": batch_id,
            "source_path": str(source),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "patient_json": json.dumps(index.patient, separators=(",", ":")),
            "ingested_at": None,  # filled per batch
            "schema_version": SCHEMA_VERSION,
        }
    )
    for c in index.concepts.values():
        for e in c.events:
            rows["events"].append(
                {
                    "patient_id": pid,
                    "concept_id": c.concept_id,
                    "kind": c.kind,
                    "display": c.display,
                    "system": c.system,
                    "code": c.code,
                    "category": c.category,
                    "date": e.date,
                    "resource_id": e.resource_id,
                    "resource_type": e.resource_type,
                    "value_num": float(e.value)
                    if isinstance(e.value, (int, float))
                    and not isinstance(e.value, bool)
                    else None,
                    "value_json": None if e.value is None else json.dumps(e.value),
                    "unit": e.unit,
                    "status": e.status,
                    "end": e.end,
                    "encounter_id": e.encounter_id,
                    "detail": e.detail,
                }
            )
    for chunk in index.notes.values():
        for i, (visit, date, ids) in enumerate(chunk.occurrences):
            rows["notes"].append(
                {
                    "patient_id": pid,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "text": chunk.text,
                    "ordinal": i,
                    "visit": visit,
                    "date": date,
                    "resource_ids": list(ids),
                }
            )
    rows["resources"] = [
        {"patient_id": pid, "resource_id": rid, "resource_type": typ}
        for rid, typ in index.resource_types.items()
    ]
    return rows


def _ingest_batch(store: str, files: list[str]) -> dict[str, Any]:
    """Worker: index a batch of bundles and write one Parquet file per table."""
    started = time.perf_counter()
    batch_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    now = pa.scalar(time.time_ns() // 1000, pa.timestamp("us", tz="UTC"))
    columns: dict[str, list[dict]] = {t: [] for t in TABLES}
    snapshots: list[tuple[PatientIndex, dict[str, Any]]] = []
    nbytes = 0
    for f in files:
        path = Path(f)
        index = PatientIndex(load_bundle(path))
        for table, rows in index_rows(index, path, batch_id).items():
            columns[table].extend(rows)
        snapshots.append((index, {"batch_id": batch_id, "source_path": str(path)}))
        nbytes += path.stat().st_size
    for row in columns["manifest"]:
        row["ingested_at"] = now.as_py()
    # The manifest is written last: a batch is visible only once all its
    # data files exist, so a crashed worker leaves orphans, never half a patient.
    for table in (*TABLES[1:], "manifest"):
        out = Path(store) / table / f"batch-{batch_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        pq.write_table(
            pa.Table.from_pylist(columns[table], schema=SCHEMAS[table]),
            tmp,
            compression="zstd",
            row_group_size=100_000,
        )
        os.replace(tmp, out)
    # Serving snapshots follow the committed batch: one object per patient,
    # overwritten by any later ingest of the same patient.
    snapshot_store = SnapshotStore(Path(store) / "snapshots")
    for index, meta in snapshots:
        snapshot_store.put(index, meta)
    return {
        "batch_id": batch_id,
        "patients": len(files),
        "bytes": nbytes,
        "events": len(columns["events"]),
        "seconds": time.perf_counter() - started,
    }


def _batches(files: list[Path], batch_bytes: int) -> Iterator[list[str]]:
    batch: list[str] = []
    size = 0
    for f in files:
        batch.append(str(f))
        size += f.stat().st_size
        if size >= batch_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def ingest(
    files: Iterable[Path],
    store: Path,
    workers: int | None = None,
    batch_mb: int = 256,
) -> dict[str, Any]:
    """Ingest bundles not yet in the store (same path, size and mtime are skipped)."""
    started = time.perf_counter()
    files = sorted(files)
    seen = _ingested_sources(store)
    todo = [
        f for f in files if (str(f), f.stat().st_size, f.stat().st_mtime_ns) not in seen
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_ingest_batch, str(store), b)
            for b in _batches(todo, batch_mb * 2**20)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    seconds = time.perf_counter() - started
    nbytes = sum(r["bytes"] for r in results)
    return {
        "files_seen": len(files),
        "files_skipped": len(files) - len(todo),
        "patients": sum(r["patients"] for r in results),
        "batches": len(results),
        "events": sum(r["events"] for r in results),
        "input_mb": round(nbytes / 2**20, 1),
        "seconds": round(seconds, 2),
        "mb_per_second": round(nbytes / 2**20 / seconds, 1) if seconds else None,
        "store_mb": round(_dir_bytes(store) / 2**20, 1),
    }


def _ingested_sources(store: Path) -> set[tuple[str, int, int]]:
    if not any((store / "manifest").glob("*.parquet")):
        return set()
    with duckdb.connect() as con:
        return set(
            con.execute(
                "SELECT source_path, source_size, source_mtime_ns FROM read_parquet(?)",
                [str(store / "manifest" / "*.parquet")],
            ).fetchall()
        )


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# -- serve ------------------------------------------------------------------


class Store:
    """Read side: current patient rows and corpus-wide SQL over the Parquet files."""

    def __init__(self, path: Path | str):
        self.path = str(path).rstrip("/")
        self.con = duckdb.connect()
        self.con.execute(
            f"""
            CREATE VIEW manifest AS
              SELECT * FROM read_parquet('{self.path}/manifest/*.parquet')
              QUALIFY row_number() OVER (PARTITION BY patient_id ORDER BY ingested_at DESC, batch_id DESC) = 1
            """
        )
        # Corpus-wide views keep only each patient's current batch.
        for table in TABLES[1:]:
            self.con.execute(
                f"""
                CREATE VIEW {table} AS
                  SELECT t.* FROM read_parquet('{self.path}/{table}/*.parquet', filename = true) t
                  JOIN manifest m USING (patient_id)
                  WHERE t.filename LIKE '%batch-' || m.batch_id || '.parquet'
                """
            )

    def patients(self) -> list[str]:
        return [
            r[0]
            for r in self.con.execute(
                "SELECT patient_id FROM manifest ORDER BY 1"
            ).fetchall()
        ]

    def patient_for_source(self, source: Path | str) -> str:
        found = self.con.execute(
            "SELECT patient_id FROM manifest WHERE source_path = ?", [str(source)]
        ).fetchone()
        if found is None:
            raise KeyError(f"{source} has not been ingested")
        return found[0]

    def sql(self, query: str, params: list | None = None) -> list[tuple]:
        return self.con.execute(query, params or []).fetchall()

    def load_index(self, patient_id: str) -> PatientIndex:
        """Rebuild one patient's index, reading only the batch that holds it."""
        found = self.con.execute(
            "SELECT batch_id, bundle_id, patient_json FROM manifest WHERE patient_id = ?",
            [patient_id],
        ).fetchone()
        if found is None:
            raise KeyError(f"patient {patient_id} is not in the store")
        batch_id, bundle_id, patient_json = found

        def rows(table: str, order: str) -> list[tuple]:
            file = f"{self.path}/{table}/batch-{batch_id}.parquet"
            cur = self.con.execute(
                f"SELECT * FROM read_parquet(?) WHERE patient_id = ? ORDER BY {order}",
                [file, patient_id],
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]

        concepts: dict[str, Concept] = {}
        for r in rows("events", "concept_id"):
            c = concepts.get(r["concept_id"])
            if c is None:
                c = concepts[r["concept_id"]] = Concept(
                    r["concept_id"],
                    r["kind"],
                    r["display"],
                    r["system"],
                    r["code"],
                    r["category"],
                )
            c.events.append(
                Event(
                    r["date"],
                    r["resource_id"],
                    r["resource_type"],
                    value=None
                    if r["value_json"] is None
                    else json.loads(r["value_json"]),
                    unit=r["unit"],
                    status=r["status"],
                    end=r["end"],
                    encounter_id=r["encounter_id"],
                    detail=r["detail"],
                )
            )
        notes: dict[str, NoteChunk] = {}
        for r in rows("notes", "chunk_id, ordinal"):
            chunk = notes.setdefault(
                r["chunk_id"], NoteChunk(r["chunk_id"], r["section"], r["text"])
            )
            chunk.occurrences.append((r["visit"], r["date"], tuple(r["resource_ids"])))
        types = {
            r["resource_id"]: r["resource_type"]
            for r in rows("resources", "resource_id")
        }
        return PatientIndex.from_parts(
            json.loads(patient_json), bundle_id, concepts, notes, types
        )

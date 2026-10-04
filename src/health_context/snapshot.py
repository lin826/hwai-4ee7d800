"""Prebuilt per-patient index snapshots: one compressed object per patient.

Ingest writes a snapshot of each patient's built ``PatientIndex``; serving
fetches one object by patient id instead of scanning tables, then rebuilds the
in-memory search structures (cheap: a few hundred concepts per patient).

The format is versioned gzip-compressed JSON with positional event rows, so
it needs no extra dependency and is readable by any language. The directory
layout shards by id prefix to keep directories small; the same keys work as
object-storage keys.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from .index import Concept, Event, NoteChunk, PatientIndex

FORMAT_VERSION = 1
_EVENT_FIELDS = (
    "date",
    "resource_id",
    "resource_type",
    "value",
    "unit",
    "status",
    "end",
    "encounter_id",
    "detail",
)


def dumps(index: PatientIndex, meta: dict[str, Any] | None = None) -> bytes:
    """Serialize everything ``PatientIndex.from_parts`` needs, plus free-form metadata."""
    payload = {
        "v": FORMAT_VERSION,
        "meta": meta or {},
        "patient": index.patient,
        "bundle_id": index.bundle_id,
        "concepts": [
            [
                c.concept_id,
                c.kind,
                c.display,
                c.system,
                c.code,
                c.category,
                [[getattr(e, f) for f in _EVENT_FIELDS] for e in c.events],
            ]
            for c in index.concepts.values()
        ],
        "notes": [
            [
                n.chunk_id,
                n.section,
                n.text,
                [[v, d, list(ids)] for v, d, ids in n.occurrences],
            ]
            for n in index.notes.values()
        ],
        "resource_types": index.resource_types,
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0)


def loads(data: bytes) -> tuple[PatientIndex, dict[str, Any]]:
    payload = json.loads(gzip.decompress(data))
    if payload.get("v") != FORMAT_VERSION:
        raise ValueError(f"unsupported snapshot version {payload.get('v')!r}")
    concepts = {}
    for concept_id, kind, display, system, code, category, events in payload[
        "concepts"
    ]:
        concepts[concept_id] = Concept(
            concept_id,
            kind,
            display,
            system,
            code,
            category,
            [Event(**dict(zip(_EVENT_FIELDS, row))) for row in events],
        )
    notes = {
        chunk_id: NoteChunk(
            chunk_id, section, text, [(v, d, tuple(ids)) for v, d, ids in occ]
        )
        for chunk_id, section, text, occ in payload["notes"]
    }
    index = PatientIndex.from_parts(
        payload["patient"],
        payload["bundle_id"],
        concepts,
        notes,
        payload["resource_types"],
    )
    return index, payload["meta"]


class SnapshotStore:
    """Snapshots on a filesystem path: ``<root>/<id[:2]>/<id>.json.gz``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, patient_id: str) -> Path:
        if not patient_id or "/" in patient_id or patient_id.startswith("."):
            raise KeyError(f"invalid patient id {patient_id!r}")
        return self.root / patient_id[:2] / f"{patient_id}.json.gz"

    def put(self, index: PatientIndex, meta: dict[str, Any] | None = None) -> int:
        target = self.path(index.patient_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = dumps(index, meta)
        # Write then rename so a reader never sees a partial snapshot.
        tmp = target.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return len(data)

    def get(self, patient_id: str) -> tuple[PatientIndex, dict[str, Any]]:
        try:
            data = self.path(patient_id).read_bytes()
        except FileNotFoundError:
            raise KeyError(f"no snapshot for patient {patient_id}") from None
        return loads(data)

    def patient_ids(self) -> list[str]:
        return sorted(
            p.name.removesuffix(".json.gz") for p in self.root.glob("*/*.json.gz")
        )

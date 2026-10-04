"""HTTP API over prebuilt patient snapshots and the agent.

Replicas hold no state of their own: each reads snapshots from shared storage
(``SNAPSHOT_DIR``) and keeps a bounded in-memory cache, so the service scales
horizontally behind any load balancer. Routing a patient to the same replica
improves cache hits but is not required for correctness.

Run: ``uv run --group serve uvicorn health_context.api:app --workers 4``
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .agent import answer
from .index import Concept, PatientIndex
from .llm import ChatClient
from .snapshot import SnapshotStore
from .tools import KIND_TITLES, catalog, run_tool


class IndexCache:
    """LRU of loaded indexes, refreshed when a snapshot file changes on disk."""

    def __init__(self, store: SnapshotStore, size: int):
        self.store = store
        self.size = size
        self._items: OrderedDict[str, tuple[int, PatientIndex, dict[str, Any]]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, patient_id: str) -> tuple[PatientIndex, dict[str, Any]]:
        try:
            version = self.store.path(patient_id).stat().st_mtime_ns
        except FileNotFoundError:
            raise KeyError(patient_id) from None
        with self._lock:
            cached = self._items.get(patient_id)
            if cached and cached[0] == version:
                self._items.move_to_end(patient_id)
                self.hits += 1
                return cached[1], cached[2]
            self.misses += 1
        # Load outside the lock; two concurrent misses may both load, which is harmless.
        index, meta = self.store.get(patient_id)
        with self._lock:
            self._items[patient_id] = (version, index, meta)
            self._items.move_to_end(patient_id)
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return index, meta


app = FastAPI(title="health-context", version="1")
_cache: IndexCache | None = None


def get_cache() -> IndexCache:
    global _cache
    if _cache is None:
        store = SnapshotStore(os.environ.get("SNAPSHOT_DIR", "store/snapshots"))
        _cache = IndexCache(store, int(os.environ.get("INDEX_CACHE_SIZE", "256")))
    return _cache


def get_llm() -> ChatClient:
    return ChatClient()


Cache = Annotated[IndexCache, Depends(get_cache)]


def _patient(cache: IndexCache, patient_id: str) -> tuple[PatientIndex, dict[str, Any]]:
    try:
        return cache.get(patient_id)
    except KeyError:
        raise HTTPException(404, f"unknown patient {patient_id}") from None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(8, ge=1, le=50)
    kinds: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    max_steps: int = Field(6, ge=1, le=12)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(cache: Cache) -> dict[str, Any]:
    if not cache.store.root.is_dir():
        raise HTTPException(503, f"snapshot directory {cache.store.root} not found")
    return {
        "status": "ready",
        "cached": len(cache._items),
        "hits": cache.hits,
        "misses": cache.misses,
    }


@app.get("/v1/patients/{patient_id}/catalog")
def get_catalog(patient_id: str, cache: Cache) -> dict[str, Any]:
    index, meta = _patient(cache, patient_id)
    return {
        "patient_id": patient_id,
        "catalog": catalog(index),
        "catalog_tokens": meta.get("catalog_tokens"),
        "snapshot": meta,
    }


@app.post("/v1/patients/{patient_id}/search")
def search(patient_id: str, request: SearchRequest, cache: Cache) -> dict[str, Any]:
    index, _ = _patient(cache, patient_id)
    unknown = set(request.kinds or ()) - {*KIND_TITLES, "note"}
    if unknown:
        raise HTTPException(422, f"unknown kinds: {sorted(unknown)}")
    hits = index.search(
        request.query, k=request.k, kinds=set(request.kinds) if request.kinds else None
    )
    return {
        "result": run_tool(
            index, "search_records", request.model_dump(exclude_none=True)
        ),
        "hits": [
            {
                "id": h.item_id,
                "kind": h.item.kind if isinstance(h.item, Concept) else "note",
                "name": h.item.display
                if isinstance(h.item, Concept)
                else h.item.section,
                "score": round(h.score, 3),
                "events": len(h.item.events)
                if isinstance(h.item, Concept)
                else len(h.item.occurrences),
            }
            for h in hits
        ],
    }


@app.get("/v1/patients/{patient_id}/timeline/{concept_id}")
def timeline(
    patient_id: str,
    concept_id: str,
    cache: Cache,
    since: Annotated[str | None, Query()] = None,
    until: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    index, _ = _patient(cache, patient_id)
    if concept_id not in index.concepts:
        raise HTTPException(404, f"unknown concept {concept_id}")
    args = {"concept_id": concept_id, "since": since, "until": until}
    return {"result": run_tool(index, "get_timeline", args)}


@app.post("/v1/patients/{patient_id}/ask")
def ask(
    patient_id: str,
    request: AskRequest,
    cache: Cache,
    llm: Annotated[ChatClient, Depends(get_llm)],
) -> dict[str, Any]:
    index, _ = _patient(cache, patient_id)
    started = time.perf_counter()
    try:
        result = answer(index, request.question, llm, request.max_steps)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(502, f"LLM server error: {error}") from None
    return {
        "answer": result.text,
        "steps": result.steps,
        "model": llm.model,
        "seconds": round(time.perf_counter() - started, 2),
    }

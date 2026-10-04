import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from health_context import api
from health_context.index import PatientIndex
from health_context.llm import Reply, ToolCall
from health_context.pipeline import load_bundle
from health_context.snapshot import SnapshotStore

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = next(DATA.glob("Merlene950_Marlin805_Thompson596_*.json"), None)
pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="extract data/ first (see DATA.md)"
)
PATIENT = "f5749532-3295-ad01-d9b5-932c997e7a01"
LATEST_HBA1C = "f5749532-3295-ad01-c588-daa4b0203574"


class ScriptedLLM:
    model = "scripted"

    def chat(self, messages, tools=None):
        if messages[-1]["role"] != "tool":
            return Reply(
                "", [ToolCall("c1", "get_timeline", {"concept_id": "lab:4548-4"})]
            )
        return Reply(f"6.31 % on 2025-09-29 (Observation/{LATEST_HBA1C})")


class DownLLM:
    model = "down"

    def chat(self, messages, tools=None):
        raise ConnectionRefusedError("LLM server not running")


@pytest.fixture
def setup(tmp_path):
    store = SnapshotStore(tmp_path)
    store.put(PatientIndex(load_bundle(SAMPLE)), {"batch_id": "b1"})
    cache = api.IndexCache(store, size=2)
    api.app.dependency_overrides[api.get_cache] = lambda: cache
    api.app.dependency_overrides[api.get_llm] = ScriptedLLM
    yield TestClient(api.app), cache, store
    api.app.dependency_overrides.clear()


def test_health_and_readiness(setup):
    client, _, _ = setup
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_catalog_search_and_timeline(setup):
    client, _, _ = setup
    catalog = client.get(f"/v1/patients/{PATIENT}/catalog").json()
    assert "[lab:4548-4]" in catalog["catalog"]
    assert catalog["snapshot"] == {"batch_id": "b1"}

    hits = client.post(
        f"/v1/patients/{PATIENT}/search", json={"query": "hba1c", "k": 3}
    ).json()
    assert hits["hits"][0]["id"] == "lab:4548-4"

    timeline = client.get(
        f"/v1/patients/{PATIENT}/timeline/lab:4548-4", params={"since": "2025-01-01"}
    )
    assert f"Observation/{LATEST_HBA1C}" in timeline.json()["result"]
    assert "1 of 8 events" in timeline.json()["result"]


def test_ask_runs_agent_with_tools(setup):
    client, _, _ = setup
    body = client.post(
        f"/v1/patients/{PATIENT}/ask", json={"question": "Latest HbA1c?"}
    ).json()
    assert LATEST_HBA1C in body["answer"]
    assert [s["name"] for s in body["steps"]] == ["get_timeline"]


def test_errors_map_to_http_status(setup):
    client, _, _ = setup
    assert client.get("/v1/patients/nobody/catalog").status_code == 404
    assert client.get(f"/v1/patients/{PATIENT}/timeline/lab:0000").status_code == 404
    assert (
        client.post(
            f"/v1/patients/{PATIENT}/search", json={"query": "x", "kinds": ["bogus"]}
        ).status_code
        == 422
    )
    api.app.dependency_overrides[api.get_llm] = DownLLM
    assert (
        client.post(f"/v1/patients/{PATIENT}/ask", json={"question": "q"}).status_code
        == 502
    )


def test_cache_hits_and_reloads_changed_snapshot(setup):
    client, cache, store = setup
    client.get(f"/v1/patients/{PATIENT}/catalog")
    client.get(f"/v1/patients/{PATIENT}/catalog")
    assert (cache.hits, cache.misses) == (1, 1)

    path = store.path(PATIENT)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    client.get(f"/v1/patients/{PATIENT}/catalog")
    assert cache.misses == 2

from pathlib import Path

import pytest

from health_context.index import PatientIndex
from health_context.pipeline import load_bundle
from health_context.snapshot import SnapshotStore, dumps, loads
from health_context.tools import catalog, run_tool

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = next(DATA.glob("Merlene950_Marlin805_Thompson596_*.json"), None)
pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="extract data/ first (see DATA.md)"
)


def _views(index: PatientIndex) -> list[str]:
    return [
        catalog(index),
        *(
            run_tool(index, "get_timeline", {"concept_id": c})
            for c in sorted(index.concepts)
        ),
        *(
            run_tool(index, "search_records", {"query": q})
            for q in ("hba1c", "allergic")
        ),
    ]


def test_snapshot_round_trip_is_exact():
    index = PatientIndex(load_bundle(SAMPLE))
    restored, meta = loads(dumps(index, {"batch_id": "b1"}))
    assert meta == {"batch_id": "b1"}
    assert _views(restored) == _views(index)


def test_snapshot_bytes_are_deterministic():
    index = PatientIndex(load_bundle(SAMPLE))
    assert dumps(index) == dumps(PatientIndex(load_bundle(SAMPLE)))


def test_store_put_get_and_rejects_path_ids(tmp_path):
    store = SnapshotStore(tmp_path)
    index = PatientIndex(load_bundle(SAMPLE))
    store.put(index)
    assert store.patient_ids() == [index.patient_id]
    assert store.get(index.patient_id)[0].patient_id == index.patient_id
    for bad in ("", "../etc", ".hidden"):
        with pytest.raises(KeyError):
            store.get(bad)
    with pytest.raises(KeyError):
        store.get("missing-patient")

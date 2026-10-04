import os
import shutil
from pathlib import Path

import pytest

from health_context.index import PatientIndex
from health_context.pipeline import load_bundle
from health_context.store import Store, ingest
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


@pytest.fixture
def bundle(tmp_path) -> Path:
    copy = tmp_path / "in" / SAMPLE.name
    copy.parent.mkdir()
    shutil.copy(SAMPLE, copy)
    return copy


def test_store_round_trip_matches_in_memory_index(bundle, tmp_path):
    report = ingest([bundle], tmp_path / "store", workers=1)
    assert report["patients"] == 1
    store = Store(tmp_path / "store")
    loaded = store.load_index(store.patient_for_source(bundle))
    assert _views(loaded) == _views(PatientIndex(load_bundle(bundle)))


def test_ingest_skips_unchanged_and_reingests_changed(bundle, tmp_path):
    store_dir = tmp_path / "store"
    ingest([bundle], store_dir, workers=1)
    assert ingest([bundle], store_dir, workers=1)["files_skipped"] == 1

    stat = bundle.stat()
    os.utime(bundle, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert ingest([bundle], store_dir, workers=1)["patients"] == 1

    store = Store(store_dir)
    events_now = store.sql("SELECT count(*) FROM events")[0][0]
    raw_rows = store.sql(
        f"SELECT count(*) FROM read_parquet('{store_dir}/events/*.parquet')"
    )[0][0]
    # The superseded batch stays on disk but is hidden from the current view.
    assert store.patients() == [PatientIndex(load_bundle(bundle)).patient_id]
    assert raw_rows == 2 * events_now


def test_unknown_patient_raises(bundle, tmp_path):
    ingest([bundle], tmp_path / "store", workers=1)
    with pytest.raises(KeyError):
        Store(tmp_path / "store").load_index("no-such-patient")

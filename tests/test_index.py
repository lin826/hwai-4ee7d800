from pathlib import Path

import pytest

from health_context.index import PatientIndex, tokenize
from health_context.pipeline import load_bundle
from health_context.tools import catalog, run_tool

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = next(DATA.glob("Merlene950_Marlin805_Thompson596_*.json"), None)
pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="extract data/ first (see DATA.md)"
)

LATEST_HBA1C = "f5749532-3295-ad01-c588-daa4b0203574"


@pytest.fixture(scope="module")
def index() -> PatientIndex:
    return PatientIndex(load_bundle(SAMPLE))


def test_lab_series_keeps_every_reading_in_date_order(index):
    a1c = index.concepts["lab:4548-4"]
    assert len(a1c.events) == 8
    assert [e.date for e in a1c.events] == sorted(e.date for e in a1c.events)
    assert (a1c.latest.value, a1c.latest.unit, a1c.latest.date[:10]) == (
        6.31,
        "%",
        "2025-09-29",
    )
    assert a1c.latest.resource_id == LATEST_HBA1C


def test_blood_pressure_components_are_separate_series(index):
    assert index.concepts["lab:8480-6"].display == "Systolic Blood Pressure"
    assert index.concepts["lab:8462-4"].display == "Diastolic Blood Pressure"
    assert "lab:85354-9" not in index.concepts


def test_medication_references_resolve_to_drug_codes(index):
    assert "medication:unknown" not in index.concepts
    assert "medication:807283" in index.concepts


def test_missing_allergies_are_not_recorded_rather_than_none(index):
    text = catalog(index)
    assert "No AllergyIntolerance resources are recorded" in text
    assert "not confirmed absent" in text
    assert not index.search("allergy", kinds={"allergy"})


def test_duplicate_note_payloads_merge_into_one_occurrence(index):
    chunk = next(c for c in index.notes.values() if c.section == "allergies")
    assert len(chunk.occurrences) == index.counts["Encounter"]
    types = {
        index.resource_types[rid] for _, _, ids in chunk.occurrences for rid in ids
    }
    assert types == {"DocumentReference", "DiagnosticReport"}


def test_search_ranks_named_concept_first(index):
    assert index.search("most recent HbA1c", k=1)[0].item_id == "lab:4548-4"
    assert (
        index.search("when was prediabetes diagnosed", k=1)[0].item_id
        == "condition:714628002"
    )


def test_timeline_cites_every_reading(index):
    out = run_tool(index, "get_timeline", {"concept_id": "lab:4548-4"})
    assert f"Observation/{LATEST_HBA1C}" in out
    assert out.count("Observation/") == 8


def test_tokenize_folds_plurals_and_lemmas():
    assert tokenize("Allergies") == tokenize("allergic") == ["allergy"]

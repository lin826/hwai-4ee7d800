from pathlib import Path

import pytest

from health_context.index import PatientIndex
from health_context.pipeline import load_bundle
from health_context.rag_eval import generate_cases, hand_cases, score_case

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = next(DATA.glob("Merlene950_Marlin805_Thompson596_*.json"), None)
pytestmark = pytest.mark.skipif(
    SAMPLE is None, reason="extract data/ first (see DATA.md)"
)


@pytest.fixture(scope="module")
def index() -> PatientIndex:
    return PatientIndex(load_bundle(SAMPLE))


def test_generated_cases_pass_on_sample(index):
    cases = generate_cases(SAMPLE, per_category=5)
    assert cases
    assert all(score_case(index, case).answer_at_1 for case in cases)


def test_hand_cases_only_apply_to_the_readme_patient(index):
    assert hand_cases(SAMPLE)
    assert hand_cases(Path("Someone_Else_123.json")) == []
    assert score_case(index, hand_cases(SAMPLE)[0]).rank == 1

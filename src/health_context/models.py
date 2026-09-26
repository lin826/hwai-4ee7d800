"""Shared data contracts for the health-context pipeline.

Keep these records plain and serializable. They preserve source identity and
clinical date semantics across context generation, retrieval, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawResource:
    resource_type: str
    resource_id: str
    data: dict[str, Any]


@dataclass(frozen=True)
class Bundle:
    bundle_id: str
    path: str
    resources: tuple[RawResource, ...]


@dataclass(frozen=True)
class SourceRef:
    bundle_id: str
    resource_type: str
    resource_id: str
    text: str | None = None
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class ClinicalDate:
    kind: str
    value: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    kind: str
    value: Any
    status: str | None = None
    dates: tuple[ClinicalDate, ...] = ()
    codes: tuple[dict[str, str], ...] = ()
    raw_value: Any = None
    units: str | None = None
    raw_unit: str | None = None
    sources: tuple[SourceRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    facts: tuple[Fact, ...]
    raw_resources: tuple[RawResource, ...]
    bundle_id: str


@dataclass(frozen=True)
class ContextArtifact:
    text: str
    selected_fact_ids: tuple[str, ...]
    omitted_fact_ids: tuple[str, ...]
    token_count: int | None
    count_method: str
    budget: int
    budget_satisfied: bool


@dataclass(frozen=True)
class Evidence:
    fact: Fact | None
    text: str
    sources: tuple[SourceRef, ...]
    dates: tuple[ClinicalDate, ...] = ()
    score: float = 0.0
    match_type: str = ""


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    patient_id: str
    question: str
    expected: tuple[str, ...]
    required_source_ids: tuple[str, ...] = ()
    allow_retrieval: bool = True
    coverage_expected: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    answer_correct: bool
    evidence_correct: bool
    budget_compliant: bool
    details: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]

    @property
    def answer_accuracy(self) -> float:
        return sum(result.answer_correct for result in self.results) / len(self.results) if self.results else 0.0

    @property
    def evidence_accuracy(self) -> float:
        return sum(result.evidence_correct for result in self.results) / len(self.results) if self.results else 0.0

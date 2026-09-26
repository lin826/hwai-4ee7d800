"""Small, provider-independent evaluation helpers for the demo pipeline.

The built-in cases are intentionally hand-checkable. They measure answer
coverage, source fidelity, and the context budget independently; they do not
claim clinical validity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .models import (
    CaseResult,
    ContextArtifact,
    EvalCase,
    Evidence,
    EvaluationReport,
    PatientRecord,
)


SAMPLE_PATIENT_ID = "f5749532-3295-ad01-d9b5-932c997e7a01"
LATEST_HBA1C_ID = "f5749532-3295-ad01-c588-daa4b0203574"
PREDIABETES_CONDITION_ID_SUFFIX = "fdfb-caeae3913925"


DEFAULT_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="hba1c_latest_exact",
        patient_id=SAMPLE_PATIENT_ID,
        question="What was her most recent HbA1c, and when was it taken?",
        expected=("6.31%", "2025-09-29"),
        required_source_ids=(LATEST_HBA1C_ID,),
    ),
    EvalCase(
        case_id="hba1c_trend_and_diagnosis",
        patient_id=SAMPLE_PATIENT_ID,
        question="Is her blood sugar getting worse? Tie the trend to a related diagnosis.",
        # A supplied answer is scored against the high-level conclusion;
        # without one, the fallback only checks source-fact availability.
        expected=("rising overall", "prediabetes"),
        required_source_ids=(LATEST_HBA1C_ID, PREDIABETES_CONDITION_ID_SUFFIX),
        coverage_expected=("6.31%", "prediabetes"),
    ),
    EvalCase(
        case_id="allergy_not_recorded",
        patient_id=SAMPLE_PATIENT_ID,
        question="What is she allergic to?",
        expected=("not recorded",),
        allow_retrieval=True,
    ),
    EvalCase(
        case_id="hba1c_source_fidelity",
        patient_id=SAMPLE_PATIENT_ID,
        question="Show the evidence for the latest HbA1c result.",
        expected=("6.31%", "2025-09-29"),
        required_source_ids=(LATEST_HBA1C_ID,),
    ),
)


Retriever = Callable[[str], Iterable[Evidence]]


def evaluate(
    record: PatientRecord,
    context: ContextArtifact,
    retriever: Retriever | None,
    cases: Iterable[EvalCase] = DEFAULT_EVAL_CASES,
    *,
    answers: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """Score cases without invoking a model or external provider.

    If ``answers`` is supplied, answer scoring checks those answer strings.
    Otherwise it measures whether the expected literals are available in the
    compact context or retrieved evidence (answerable-fact coverage). Retrieval
    evidence is used for evidence scoring in both modes.

    ``retriever`` is a query-to-evidence callable, matching the architecture
    contract. Evidence sources must belong to this patient bundle and resolve
    to a resource in ``record``. A count is budget-compliant only when it is
    within budget and its method identifies Anthropic token counting.
    """

    all_facts = {fact.fact_id: fact for fact in record.facts}
    resource_ids = {resource.resource_id for resource in record.raw_resources}
    resource_types_by_id = {
        resource.resource_id: resource.resource_type for resource in record.raw_resources
    }
    context_text = context.text.casefold()
    within_authoritative_budget = (
        context.token_count is not None
        and context.token_count <= context.budget
        and "anthropic" in context.count_method.casefold()
    )

    results: list[CaseResult] = []
    for case in cases:
        evidence: tuple[Evidence, ...] = ()
        if retriever is not None and case.allow_retrieval:
            evidence = tuple(retriever(case.question))

        evidence_text = "\n".join(_evidence_text(item) for item in evidence).casefold()
        answer_text = (answers or {}).get(case.case_id)
        available_text = context_text + "\n" + evidence_text
        scored_text = (
            answer_text.casefold()
            if answer_text is not None
            else available_text
        )
        literals = case.expected if answer_text is not None else (case.coverage_expected or case.expected)

        expected_present = all(expected.casefold() in scored_text for expected in literals)
        if case.case_id == "allergy_not_recorded":
            has_allergy_resource = any(
                resource.resource_type == "AllergyIntolerance"
                for resource in record.raw_resources
            )
            # Negative evidence is valid only as a statement about this bundle's
            # contents, never as a claim that the patient has no allergies.
            no_record_wording = any(
                phrase in scored_text
                for phrase in ("not recorded", "no allergyintolerance", "no allergy record")
            )
            answer_correct = not has_allergy_resource and no_record_wording
            evidence_correct = (
                not has_allergy_resource
                and all(
                    source.resource_type != "AllergyIntolerance"
                    for item in evidence
                    for source in item.sources
                )
            )
        else:
            answer_correct = expected_present
            evidence_correct = _required_sources_present(
                case.required_source_ids,
                evidence,
                context,
                all_facts,
                record,
                resource_ids,
                resource_types_by_id,
            )

        evidence_valid = all(
            _source_belongs_to_record(source, record, resource_ids)
            for item in evidence
            for source in item.sources
        )
        evidence_correct = evidence_correct and evidence_valid
        details = (
            "answer measured from supplied answer"
            if answer_text is not None
            else "answer measured as context/retrieval fact coverage"
        )
        if not within_authoritative_budget:
            details += "; context lacks an in-budget authoritative Anthropic count"
        results.append(
            CaseResult(
                case_id=case.case_id,
                answer_correct=answer_correct,
                evidence_correct=evidence_correct,
                budget_compliant=within_authoritative_budget,
                details=details,
            )
        )
    return EvaluationReport(results=tuple(results))


def _required_sources_present(
    required_ids: tuple[str, ...],
    evidence: tuple[Evidence, ...],
    context: ContextArtifact,
    facts: dict[str, Any],
    record: PatientRecord,
    resource_ids: set[str],
    resource_types_by_id: dict[str, str],
) -> bool:
    if not required_ids:
        return True

    observed_ids = {
        source.resource_id
        for item in evidence
        for source in item.sources
        if _source_belongs_to_record(source, record, resource_ids)
    }
    # selected_fact_ids refer to fact IDs; map those back to their sources.
    for fact_id in context.selected_fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            if not all(
                _source_belongs_to_record(source, record, resource_ids)
                for source in fact.sources
            ):
                return False
            observed_ids.update(source.resource_id for source in fact.sources)

    def present(required: str) -> bool:
        return any(
            observed == required or observed.endswith(required)
            for observed in observed_ids
        )

    # Reject a malformed source that names an existing ID with the wrong type.
    for item in evidence:
        for source in item.sources:
            if source.resource_id in resource_ids:
                expected_type = resource_types_by_id[source.resource_id]
                if source.resource_type != expected_type:
                    return False
    return all(present(source_id) for source_id in required_ids)


def _source_belongs_to_record(source: Any, record: PatientRecord, resource_ids: set[str]) -> bool:
    return (
        source.bundle_id == record.bundle_id
        and source.resource_id in resource_ids
    )


def _evidence_text(evidence: Evidence) -> str:
    """Render structured evidence fields for literal answer coverage checks."""
    parts = [evidence.text]
    if evidence.fact is not None:
        parts.extend((str(evidence.fact.value), str(evidence.fact.raw_value or "")))
        if evidence.fact.units:
            parts.append(evidence.fact.units)
    parts.extend(date.value for date in evidence.dates)
    if evidence.fact is not None:
        parts.extend(date.value for date in evidence.fact.dates)
    return " ".join(parts)

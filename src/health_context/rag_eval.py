"""Retrieval eval: questions with known answers, generated from raw FHIR.

Ground truth is computed directly from the bundle JSON with deliberately
simple code that shares nothing with ``index.py``. A case passes retrieval@k
when the resource holding the answer is inside one of the top-k items, and
passes answer@1 when the top item's own first/latest event reproduces the
ground-truth date (and value, for labs). No model is involved, so the numbers
isolate retrieval quality from answer generation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .index import Concept, PatientIndex
from .pipeline import load_bundle


@dataclass(frozen=True)
class RetrievalCase:
    patient_file: str
    category: str
    question: str
    gold_resource_id: str
    gold_date: str
    gold_value: Any = None
    # Which event of the top concept should hold the answer.
    position: str = "latest"


_TAG = re.compile(
    r"\s*\((disorder|finding|procedure|situation|regime/therapy|observable entity|person|qualifier value|physical object)\)\s*$"
)


def _name(concept: dict[str, Any]) -> str:
    coding = (concept.get("coding") or [{}])[0]
    return _TAG.sub("", concept.get("text") or coding.get("display", "")).strip()


def _code(concept: dict[str, Any]) -> str:
    return (concept.get("coding") or [{}])[0].get("code", "")


def _pick(items: list[str], n: int, salt: str) -> list[str]:
    """Deterministic pseudo-random sample, stable across runs and machines."""
    return sorted(
        items, key=lambda x: hashlib.sha256(f"{salt}:{x}".encode()).hexdigest()
    )[:n]


def generate_cases(path: str | Path, per_category: int = 3) -> list[RetrievalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    resources = [e["resource"] for e in data.get("entry", [])]
    name = Path(path).name
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in resources:
        t = r["resourceType"]
        if t == "Observation" and isinstance(r.get("valueQuantity"), dict):
            groups["lab"][_code(r["code"])].append(r)
        elif t == "Condition":
            groups["condition"][_code(r["code"])].append(r)
        elif t == "MedicationRequest" and "medicationCodeableConcept" in r:
            groups["medication"][_code(r["medicationCodeableConcept"])].append(r)
        elif t == "Immunization":
            groups["immunization"][_code(r["vaccineCode"])].append(r)
        elif t == "Procedure":
            groups["procedure"][_code(r["code"])].append(r)

    dates = {
        "lab": lambda r: r.get("effectiveDateTime", ""),
        "condition": lambda r: r.get("onsetDateTime", ""),
        "medication": lambda r: r.get("authoredOn", ""),
        "immunization": lambda r: r.get("occurrenceDateTime", ""),
        "procedure": lambda r: (
            r.get("performedDateTime")
            or (r.get("performedPeriod") or {}).get("start", "")
        ),
    }
    concept_field = {
        "lab": "code",
        "condition": "code",
        "medication": "medicationCodeableConcept",
        "immunization": "vaccineCode",
        "procedure": "code",
    }
    cases: list[RetrievalCase] = []
    for category, by_code in groups.items():
        # Labs need at least two readings so "most recent" is a real choice.
        eligible = [
            c for c, rs in by_code.items() if c and (category != "lab" or len(rs) >= 2)
        ]
        for code in _pick(eligible, per_category, f"{name}:{category}"):
            rs = sorted(by_code[code], key=lambda r: (dates[category](r), r["id"]))
            label = _name(rs[0][concept_field[category]])
            if category == "condition":
                gold, position = rs[0], "first"
                question = f"When was {label.lower()} first diagnosed?"
            else:
                gold, position = rs[-1], "latest"
                question = {
                    "lab": f"What was the most recent {label} result and when was it measured?",
                    "medication": f"When was {label} last prescribed?",
                    "immunization": f"When did the patient last receive the {label} vaccine?",
                    "procedure": f"When was the most recent {label.lower()}?",
                }[category]
            value = gold["valueQuantity"].get("value") if category == "lab" else None
            cases.append(
                RetrievalCase(
                    name,
                    f"{category}_{position}",
                    question,
                    gold["id"],
                    dates[category](gold)[:10],
                    value,
                    position,
                )
            )
    return cases


# Natural-language paraphrases for the README's sample patient; templated
# questions above reuse coded names, these do not.
SAMPLE_FILE_PREFIX = "Merlene950_Marlin805_Thompson596_"
HAND_CASES = (
    (
        "hand_lab_latest",
        "What was her most recent HbA1c, and when was it taken?",
        "f5749532-3295-ad01-c588-daa4b0203574",
        "2025-09-29",
        6.31,
        "latest",
    ),
    (
        "hand_lab_latest",
        "Is her blood sugar getting worse?",
        "f5749532-3295-ad01-c588-daa4b0203574",
        "2025-09-29",
        6.31,
        "latest",
    ),
    (
        "hand_lab_latest",
        "What is her latest A1c?",
        "f5749532-3295-ad01-c588-daa4b0203574",
        "2025-09-29",
        6.31,
        "latest",
    ),
)


def hand_cases(path: str | Path) -> list[RetrievalCase]:
    name = Path(path).name
    if not name.startswith(SAMPLE_FILE_PREFIX):
        return []
    return [
        RetrievalCase(name, cat, q, rid, d, v, pos)
        for cat, q, rid, d, v, pos in HAND_CASES
    ]


@dataclass(frozen=True)
class CaseOutcome:
    case: RetrievalCase
    rank: int | None  # 1-based rank of the first item containing the gold resource
    answer_at_1: bool


def score_case(index: PatientIndex, case: RetrievalCase, k: int = 5) -> CaseOutcome:
    hits = index.search(case.question, k=k)
    rank = next(
        (
            i + 1
            for i, h in enumerate(hits)
            if case.gold_resource_id in h.item.resource_ids
        ),
        None,
    )
    answer = False
    if hits and isinstance(hits[0].item, Concept):
        event = hits[0].item.latest if case.position == "latest" else hits[0].item.first
        answer = (
            event is not None
            and event.resource_id == case.gold_resource_id
            and (event.date or "")[:10] == case.gold_date
            and (case.gold_value is None or event.value == case.gold_value)
        )
    return CaseOutcome(case, rank, answer)


def run(
    paths: list[Path],
    per_category: int = 3,
    k: int = 5,
    index_for: Callable[[Path], PatientIndex] | None = None,
) -> dict[str, Any]:
    """Score every generated case; ``index_for`` swaps in a store-backed index."""
    outcomes: list[CaseOutcome] = []
    started = time.perf_counter()
    for path in paths:
        index = index_for(path) if index_for else PatientIndex(load_bundle(path))
        for case in [*generate_cases(path, per_category), *hand_cases(path)]:
            outcomes.append(score_case(index, case, k))
    by_cat: dict[str, list[CaseOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.case.category].append(o)
    by_cat["ALL"] = outcomes

    def summary(items: list[CaseOutcome]) -> dict[str, float | int]:
        n = len(items)
        return {
            "n": n,
            "recall@1": sum(o.rank == 1 for o in items) / n,
            f"recall@{k}": sum(o.rank is not None for o in items) / n,
            "mrr": sum(1 / o.rank for o in items if o.rank) / n,
            "answer@1": sum(o.answer_at_1 for o in items) / n,
        }

    return {
        "patients": len(paths),
        "seconds": round(time.perf_counter() - started, 1),
        "metrics": {cat: summary(items) for cat, items in sorted(by_cat.items())},
        "failures": [
            {
                "file": o.case.patient_file,
                "category": o.case.category,
                "question": o.case.question,
                "gold": o.case.gold_resource_id,
                "rank": o.rank,
            }
            for o in outcomes
            if o.rank != 1
        ],
    }


def format_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    cols = list(next(iter(metrics.values())).keys())
    lines = [
        f"{report['patients']} patients, {report['seconds']}s",
        "category".ljust(22) + "".join(c.rjust(11) for c in cols),
    ]
    for cat, row in metrics.items():
        lines.append(
            cat.ljust(22)
            + "".join(
                (f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(11)
                for v in row.values()
            )
        )
    return "\n".join(lines)

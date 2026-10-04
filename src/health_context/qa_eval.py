"""End-to-end eval: a model answers known-answer questions using the agent.

Grading is deterministic. An answer is correct when it states the ground-truth
date (and value, for labs) or, for absence questions, says the fact is not
recorded without inventing one. Citation is scored separately: did the answer
name the ground-truth FHIR resource id? Ground truth comes from ``rag_eval``,
which reads raw bundle JSON independently of the index.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import answer
from .index import PatientIndex
from .llm import ChatClient
from .pipeline import load_bundle
from .rag_eval import _TAG, generate_cases, hand_cases

NOT_RECORDED = re.compile(
    r"not recorded|no record|no (known )?allerg|none (are )?recorded|no .* (is|are) recorded|not (found|documented|present) in",
    re.IGNORECASE,
)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class QACase:
    patient_file: str
    category: str
    question: str
    expect_date: str | None = None
    expect_value: Any = None
    expect_text: str | None = None
    expect_absent: bool = False
    gold_resource_id: str | None = None


def build_cases(paths: list[Path], per_category: int = 1) -> list[QACase]:
    cases: list[QACase] = []
    conditions_by_file: dict[str, set[str]] = {}
    raw_by_file: dict[str, list[dict]] = {}
    for path in paths:
        resources = [e["resource"] for e in json.loads(path.read_text())["entry"]]
        raw_by_file[path.name] = resources
        conditions_by_file[path.name] = {
            _TAG.sub("", r["code"]["coding"][0]["display"]).strip()
            for r in resources
            if r["resourceType"] == "Condition"
        }
        for c in generate_cases(path, per_category):
            cases.append(
                QACase(
                    c.patient_file,
                    c.category,
                    c.question,
                    expect_date=c.gold_date,
                    expect_value=c.gold_value,
                    gold_resource_id=c.gold_resource_id,
                )
            )
        for c in hand_cases(path):
            cases.append(
                QACase(
                    c.patient_file,
                    c.category,
                    c.question,
                    expect_date=c.gold_date,
                    expect_value=c.gold_value,
                    gold_resource_id=c.gold_resource_id,
                )
            )
    all_conditions = (
        sorted(set().union(*conditions_by_file.values())) if conditions_by_file else []
    )
    for path in paths:
        resources = raw_by_file[path.name]
        allergies = [r for r in resources if r["resourceType"] == "AllergyIntolerance"]
        if allergies:
            first = allergies[0]
            cases.append(
                QACase(
                    path.name,
                    "allergy_present",
                    "What is the patient allergic to?",
                    expect_text=_TAG.sub("", first["code"]["coding"][0]["display"])
                    .strip()
                    .lower(),
                    gold_resource_id=first["id"],
                )
            )
        else:
            cases.append(
                QACase(
                    path.name,
                    "allergy_absent",
                    "What is the patient allergic to?",
                    expect_absent=True,
                )
            )
        missing = [c for c in all_conditions if c not in conditions_by_file[path.name]]
        if missing:
            pick = min(
                missing,
                key=lambda c: hashlib.sha256(f"{path.name}:{c}".encode()).hexdigest(),
            )
            cases.append(
                QACase(
                    path.name,
                    "condition_absent",
                    f"When was the patient diagnosed with {pick.lower()}?",
                    expect_absent=True,
                )
            )
    return cases


def grade(case: QACase, text: str) -> tuple[bool, bool]:
    """Return (answer_correct, cited_gold_resource)."""
    lowered = text.lower()
    cited = bool(case.gold_resource_id) and case.gold_resource_id in text
    if case.expect_absent:
        # An absence answer must not assert a date of diagnosis or allergen.
        return bool(NOT_RECORDED.search(text)) and not ISO_DATE.search(text), cited
    ok = True
    if case.expect_date:
        ok &= case.expect_date in text
    if case.expect_value is not None:
        v = case.expect_value
        forms = {str(v), f"{v:.2f}", f"{v:.1f}"} if isinstance(v, float) else {str(v)}
        ok &= any(f in text for f in forms)
    if case.expect_text:
        ok &= case.expect_text in lowered
    return ok, cited


def run(
    paths: list[Path],
    client: ChatClient,
    per_category: int = 1,
    index_for: Callable[[Path], PatientIndex] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    cases = build_cases(paths, per_category)
    by_file = {p.name: p for p in paths}
    indexes: dict[str, PatientIndex] = {}
    rows = []
    for i, case in enumerate(cases, 1):
        path = by_file[case.patient_file]
        if case.patient_file not in indexes:
            indexes[case.patient_file] = (
                index_for(path) if index_for else PatientIndex(load_bundle(path))
            )
        started = time.perf_counter()
        try:
            result = answer(indexes[case.patient_file], case.question, client)
            text, steps, error = result.text, result.steps, None
        except (
            OSError,
            ValueError,
            KeyError,
        ) as exc:  # server or parse failure fails the case
            text, steps, error = "", [], repr(exc)
        seconds = time.perf_counter() - started
        correct, cited = grade(case, text)
        rows.append(
            {
                **asdict(case),
                "answer": text,
                "correct": correct,
                "cited": cited,
                "tool_calls": len(steps),
                "steps": steps,
                "seconds": round(seconds, 2),
                "error": error,
            }
        )
        if progress:
            progress(
                f"[{i}/{len(cases)}] {'PASS' if correct else 'FAIL'} {case.category}: {case.question[:70]}"
            )
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    by_cat["ALL"] = rows

    def summary(items: list[dict]) -> dict[str, float | int]:
        n = len(items)
        cite_items = [r for r in items if r["gold_resource_id"]]
        return {
            "n": n,
            "correct": sum(r["correct"] for r in items) / n,
            "cited": sum(r["cited"] for r in cite_items) / len(cite_items)
            if cite_items
            else float("nan"),
            "tools/q": sum(r["tool_calls"] for r in items) / n,
            "sec/q": sum(r["seconds"] for r in items) / n,
            "errors": sum(r["error"] is not None for r in items),
        }

    return {
        "model": client.model,
        "base_url": client.base_url,
        "patients": len(paths),
        "seconds": round(time.perf_counter() - run_started, 1),
        "metrics": {c: summary(items) for c, items in sorted(by_cat.items())},
        "cases": rows,
    }

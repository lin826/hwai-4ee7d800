"""Deterministic, patient-scoped lexical retrieval over normalized facts.

This module deliberately does not infer clinical meaning. It ranks records by
query term overlap and returns the original fact or raw narrative text with its
FHIR resource provenance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .models import Evidence, Fact, PatientRecord, SourceRef

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_NARRATIVE_KEYS = {"text", "div", "note", "annotation", "display", "title", "description"}
_SKIP_KEYS = {"data", "url", "id", "reference", "identifier", "meta", "extension"}


def retrieve(record: PatientRecord, query: str, limit: int = 10) -> list[Evidence]:
    """Search one patient's facts and embedded narrative text.

    Matching is case-insensitive lexical overlap. Results are ordered by query
    coverage, phrase match, then stable source/fact identity. A result is always
    scoped to ``record``; the function accepts no external patient identifier
    or corpus to accidentally broaden that scope.
    """
    if limit <= 0 or not query.strip():
        return []

    terms = _terms(query)
    if not terms:
        return []
    phrase = " ".join(terms)
    candidates: list[tuple[Evidence, str, str]] = []

    for fact in record.facts:
        # Facts belong to this PatientRecord; reject malformed cross-bundle refs
        # rather than surfacing them under the wrong patient.
        sources = tuple(source for source in fact.sources if source.bundle_id == record.bundle_id)
        if not sources:
            continue
        rendered = _render_fact(fact)
        candidates.append((_evidence_for_fact(fact, rendered, sources), rendered, fact.fact_id))

    for resource in record.raw_resources:
        narrative = _extract_narrative(resource.data)
        if not narrative:
            continue
        source = SourceRef(
            bundle_id=record.bundle_id,
            resource_type=resource.resource_type,
            resource_id=resource.resource_id,
            text=narrative,
        )
        candidates.append((Evidence(
            fact=None,
            text=narrative,
            sources=(source,),
            dates=(),
            match_type="raw_narrative",
        ), narrative, f"{resource.resource_type}/{resource.resource_id}"))

    ranked: list[tuple[tuple[float, int, str, str], Evidence]] = []
    seen: set[tuple[str, str]] = set()
    for evidence, searchable, stable_id in candidates:
        normalized = _normalize(searchable)
        present = sum(1 for term in terms if term in normalized)
        if present == 0:
            continue
        coverage = present / len(terms)
        phrase_match = int(phrase in normalized)
        match_type = "phrase" if phrase_match else ("all_terms" if present == len(terms) else "terms")
        result = Evidence(
            fact=evidence.fact,
            text=evidence.text,
            sources=evidence.sources,
            dates=evidence.dates,
            score=coverage + 0.25 * phrase_match,
            match_type=match_type,
        )
        identity = (stable_id, result.text)
        if identity in seen:
            continue
        seen.add(identity)
        # Descending relevance, then deterministic stable order.
        rank_key = (-result.score, -present, stable_id, result.text)
        ranked.append((rank_key, result))

    ranked.sort(key=lambda item: item[0])
    return [evidence for _, evidence in ranked[:limit]]


def _evidence_for_fact(fact: Fact, text: str, sources: tuple[SourceRef, ...]) -> Evidence:
    return Evidence(
        fact=fact,
        text=text,
        sources=sources,
        dates=fact.dates,
        match_type="fact",
    )


def _render_fact(fact: Fact) -> str:
    """Render searchable fact fields without discarding raw values or dates."""
    parts = [fact.kind, _plain(fact.value)]
    if fact.status:
        parts.append(f"status: {fact.status}")
    if fact.raw_value is not None:
        parts.append(f"raw value: {_plain(fact.raw_value)}")
    if fact.units:
        parts.append(f"unit: {fact.units}")
    if fact.raw_unit:
        parts.append(f"raw unit: {fact.raw_unit}")
    for date in fact.dates:
        parts.append(f"{date.kind}: {date.value}")
    for code in fact.codes:
        parts.extend(str(value) for value in code.values())
    parts.extend(_plain(fact.metadata).splitlines())
    return " | ".join(part for part in parts if part)


def _extract_narrative(value: Any, parent_key: str = "") -> str:
    """Collect human-readable FHIR narrative/note fields, skipping payloads."""
    found: list[str] = []

    def visit(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                key_lower = str(child_key).casefold()
                if key_lower in _SKIP_KEYS:
                    continue
                if key_lower in _NARRATIVE_KEYS:
                    collect_text(child)
                else:
                    visit(child, key_lower)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child, key)

    def collect_text(node: Any) -> None:
        if isinstance(node, str):
            stripped = node.strip()
            if stripped:
                found.append(stripped)
        elif isinstance(node, Mapping):
            # FHIR Narrative.text.div and Annotation.text are common shapes.
            for child_key, child in node.items():
                if str(child_key).casefold() in {"div", "text", "title", "display"}:
                    collect_text(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                collect_text(child)

    visit(value, parent_key)
    # Preserve source order while removing duplicate repeated strings.
    return "\n".join(dict.fromkeys(found))


def _plain(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _terms(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token.casefold() for token in _WORD_RE.findall(text)))


def _normalize(text: str) -> str:
    return " ".join(_terms(text))

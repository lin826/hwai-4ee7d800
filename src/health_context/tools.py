"""Text views of a PatientIndex: the always-on catalog and the retrieval tools.

The catalog tells a model *what exists* (every concept, its count, date span
and latest value) in a few thousand tokens. The tools return the detail behind
a catalog line on demand, always with FHIR resource ids for citation.
"""

from __future__ import annotations

from typing import Any

from .index import Concept, Event, NoteChunk, PatientIndex

KIND_TITLES = {
    "condition": "Conditions",
    "medication": "Medications (prescriptions)",
    "allergy": "Allergies",
    "lab": "Labs, vitals and survey results",
    "immunization": "Immunizations",
    "procedure": "Procedures",
    "encounter": "Encounter types",
    "careplan": "Care plans",
    "device": "Devices",
    "imaging": "Imaging studies",
}


def _day(value: str | None) -> str:
    return value[:10] if value else "undated"


def _value(e: Event) -> str:
    if e.value is None:
        return ""
    return f"{e.value} {e.unit}".strip() if e.unit else str(e.value)


def _span(c: Concept) -> str:
    first, latest = c.first, c.latest
    if not first or not latest:
        return ""
    return (
        _day(first.date)
        if first is latest
        else f"{_day(first.date)}..{_day(latest.date)}"
    )


def concept_line(c: Concept) -> str:
    """One catalog line: id, name, count, span, and the latest state."""
    latest = c.latest
    parts = [f"[{c.concept_id}] {c.display}", f"n={len(c.events)}", _span(c)]
    if latest is not None:
        if c.kind == "lab":
            parts.append(f"latest {_value(latest)} on {_day(latest.date)}")
        elif c.kind == "condition":
            parts.append(
                "active"
                if latest.status == "active"
                else f"{latest.status}" + (f" {_day(latest.end)}" if latest.end else "")
            )
        elif c.kind in ("medication", "careplan", "allergy", "device"):
            parts.append(f"status {latest.status}")
    return "; ".join(p for p in parts if p)


def catalog(index: PatientIndex) -> str:
    """Compact table of contents for one patient, meant to sit in the prompt."""
    p = index.patient
    name = next(iter(p.get("name") or []), {})
    lines = [
        f"Patient {' '.join(name.get('given', []))} {name.get('family', '')}".strip()
        + f"; sex {p.get('gender')}; born {p.get('birthDate')}"
        + (
            f"; died {_day(p['deceasedDateTime'])}" if p.get("deceasedDateTime") else ""
        ),
        f"Patient id {index.patient_id}. Records span "
        + f"{_record_span(index)}. Concept ids in [brackets] can be passed to get_timeline.",
    ]
    for kind, title in KIND_TITLES.items():
        concepts = index.concepts_of(kind)
        if kind == "allergy" and not concepts:
            # Absence of a resource is not proof of no allergy; say exactly that.
            lines += [
                "",
                f"## {title}",
                "No AllergyIntolerance resources are recorded in this bundle (not recorded, not confirmed absent).",
            ]
            continue
        if not concepts:
            continue
        lines += ["", f"## {title} ({len(concepts)})"]
        lines += [f"- {concept_line(c)}" for c in concepts]
    return "\n".join(lines)


def _record_span(index: PatientIndex) -> str:
    dates = [e.date for c in index.concepts.values() for e in c.events if e.date]
    return f"{_day(min(dates))} to {_day(max(dates))}" if dates else "unknown dates"


def timeline(
    c: Concept, since: str | None = None, until: str | None = None, limit: int = 200
) -> str:
    """Every event of one concept with its resource id, newest last.

    Consecutive identical lab values are collapsed into one dated run so long,
    flat series stay short; the run still lists its first and last resource ids.
    """
    events = [
        e
        for e in c.events
        if (not since or (e.date or "") >= since)
        and (not until or (e.date or "")[:10] <= until)
    ]
    header = f"{c.display} [{c.concept_id}; {c.system or ''} {c.code or ''}] {len(events)} of {len(c.events)} events"
    rows: list[str] = []
    i = 0
    while i < len(events):
        e = events[i]
        j = i
        while (
            c.kind == "lab"
            and j + 1 < len(events)
            and events[j + 1].value == e.value
            and events[j + 1].unit == e.unit
        ):
            j += 1
        if j > i + 1:
            rows.append(
                f"{_day(e.date)}..{_day(events[j].date)} {_value(e)} x{j - i + 1} (Observation/{e.resource_id} .. Observation/{events[j].resource_id})"
            )
            i = j + 1
            continue
        extra = [_value(e)]
        if e.status and c.kind != "lab":
            extra.append(f"status {e.status}")
        if e.end:
            extra.append(f"until {_day(e.end)}")
        if e.detail:
            extra.append(e.detail)
        rows.append(
            f"{_day(e.date)} "
            + "; ".join(x for x in extra if x)
            + f" ({e.resource_type}/{e.resource_id})"
        )
        i += 1
    if len(rows) > limit:
        rows = [
            f"... {len(rows) - limit} earlier rows omitted; narrow with since/until"
        ] + rows[-limit:]
    return "\n".join([header, *rows])


def note_view(
    chunk: NoteChunk, resource_types: dict[str, str], max_dates: int = 5
) -> str:
    dates = sorted({_day(d) for _, d, _ in chunk.occurrences})
    shown = ", ".join(dates[-max_dates:]) + (
        f" (+{len(dates) - max_dates} earlier)" if len(dates) > max_dates else ""
    )
    latest_ids = ", ".join(
        f"{resource_types.get(rid, 'Resource')}/{rid}"
        for rid in chunk.occurrences[-1][2]
    )
    return f"[{chunk.chunk_id}] note section '{chunk.section}' in {len(chunk.occurrences)} notes ({shown}); latest source {latest_ids}\n{chunk.text}"


def search_view(
    index: PatientIndex, query: str, k: int = 8, kinds: list[str] | None = None
) -> str:
    hits = index.search(query, k=k, kinds=set(kinds) if kinds else None)
    if not hits:
        return f"No indexed record matches {query!r}. Absence here means not recorded in this bundle, not confirmed absent."
    out = []
    for h in hits:
        out.append(
            f"- {concept_line(h.item)}"
            if isinstance(h.item, Concept)
            else "- " + note_view(h.item, index.resource_types).replace("\n", "\n  ")
        )
    return "\n".join(out)


def encounter_view(index: PatientIndex, encounter_id: str) -> str:
    linked = index.by_encounter.get(encounter_id, [])
    enc = next(
        (
            c
            for c in index.concepts.values()
            for e in c.events
            if e.resource_id == encounter_id
        ),
        None,
    )
    if enc is None:
        return f"No encounter {encounter_id} in this bundle."
    event = next(e for e in enc.events if e.resource_id == encounter_id)
    lines = [
        f"Encounter/{encounter_id}: {enc.display} {_day(event.date)}"
        + (f"; {event.detail}" if event.detail else "")
    ]
    by_id = {e.resource_id: (c, e) for c in index.concepts.values() for e in c.events}
    for rid in dict.fromkeys(linked):
        if rid in by_id:
            c, e = by_id[rid]
            lines.append(
                f"- {c.kind}: {c.display} {_value(e)} ({e.resource_type}/{rid})"
            )
    return "\n".join(lines)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_records",
        "description": "Search this patient's records (labs, conditions, medications, procedures, immunizations, encounters, allergies, notes). Returns matching concepts with count, date span and latest value, and matching note sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clinical terms, e.g. 'hba1c', 'statin', 'flu vaccine'.",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": [*KIND_TITLES, "note"]},
                },
                "k": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_timeline",
        "description": "All dated events of one concept (e.g. every HbA1c reading) with values, units, statuses and FHIR resource ids. Optional ISO date bounds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "concept_id": {
                    "type": "string",
                    "description": "Id in brackets from the catalog or search, e.g. 'lab:4548-4'.",
                },
                "since": {"type": "string"},
                "until": {"type": "string"},
            },
            "required": ["concept_id"],
        },
    },
    {
        "name": "get_encounter",
        "description": "Everything recorded during one visit: conditions, labs, procedures, medications, with resource ids.",
        "input_schema": {
            "type": "object",
            "properties": {"encounter_id": {"type": "string"}},
            "required": ["encounter_id"],
        },
    },
]


def run_tool(index: PatientIndex, name: str, args: dict[str, Any]) -> str:
    if name == "search_records":
        return search_view(
            index, args["query"], int(args.get("k", 8)), args.get("kinds")
        )
    if name == "get_timeline":
        concept = index.concepts.get(args["concept_id"])
        if concept is None:
            return f"Unknown concept id {args['concept_id']!r}; use search_records to find one."
        return timeline(concept, args.get("since"), args.get("until"))
    if name == "get_encounter":
        return encounter_view(index, args["encounter_id"])
    return f"Unknown tool {name!r}."

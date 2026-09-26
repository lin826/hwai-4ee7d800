"""Deterministic ingestion and compact context generation for FHIR bundles.

This module intentionally keeps selection policy at the context boundary. It
does not interpret codes or infer clinical relationships absent from the source.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable

from .models import Bundle, ClinicalDate, ContextArtifact, Fact, PatientRecord, RawResource, SourceRef


def load_bundle(path: str | Path) -> Bundle:
    """Load a patient FHIR Bundle while preserving each original resource."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("resourceType") != "Bundle":
        raise ValueError(f"Expected a FHIR Bundle object in {source}")
    entries = document.get("entry", [])
    if not isinstance(entries, list):
        raise ValueError("FHIR Bundle.entry must be a list")
    resources: list[RawResource] = []
    for index, entry in enumerate(entries):
        raw = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(raw, dict) or not isinstance(raw.get("resourceType"), str):
            raise ValueError(f"Bundle entry {index} has no valid resource")
        resources.append(RawResource(raw["resourceType"], str(raw.get("id", "")), raw))
    patient = next((r for r in resources if r.resource_type == "Patient"), None)
    bundle_id = str(document.get("id") or (patient.resource_id if patient else source.stem))
    return Bundle(bundle_id, str(source), tuple(resources))


def _coding(concept: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(concept, dict):
        return ()
    out = []
    for coding in concept.get("coding", []) or []:
        if isinstance(coding, dict):
            item = {k: str(coding[k]) for k in ("system", "code", "display") if coding.get(k) is not None}
            if item:
                out.append(item)
    if concept.get("text"):
        out.append({"display": str(concept["text"])})
    return tuple(out)


def _dates(resource_type: str, r: dict[str, Any]) -> tuple[ClinicalDate, ...]:
    # The kind labels preserve the field's meaning; dates are never merged.
    fields: dict[str, tuple[tuple[str, str], ...]] = {
        "Patient": (("birthDate", "birth"), ("deceasedDateTime", "death")),
        "Condition": (("onsetDateTime", "onset"), ("abatementDateTime", "abatement"), ("recordedDate", "recorded")),
        "AllergyIntolerance": (("onsetDateTime", "onset"), ("recordedDate", "recorded"), ("lastOccurrence", "last_occurrence")),
        "Observation": (("effectiveDateTime", "effective"), ("effectivePeriod.start", "effective_start"), ("effectivePeriod.end", "effective_end"), ("issued", "issued")),
        "DiagnosticReport": (("effectiveDateTime", "effective"), ("issued", "issued")),
        "MedicationRequest": (("authoredOn", "authored"), ("dispenseRequest.validityPeriod.start", "validity_start"), ("dispenseRequest.validityPeriod.end", "validity_end")),
        "Encounter": (("period.start", "encounter_start"), ("period.end", "encounter_end")),
        "Procedure": (("performedDateTime", "performed"), ("performedPeriod.start", "performed_start"), ("performedPeriod.end", "performed_end")),
        "Immunization": (("occurrenceDateTime", "occurrence"), ("recorded", "recorded")),
        "DocumentReference": (("date", "document_date"),),
        "MedicationAdministration": (("effectiveDateTime", "effective"), ("effectivePeriod.start", "effective_start"), ("effectivePeriod.end", "effective_end")),
    }
    result = []
    for path, kind in fields.get(resource_type, ()):
        v: Any = r
        for part in path.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if v is not None:
            result.append(ClinicalDate(kind, str(v)))
    return tuple(result)


def _value(resource_type: str, r: dict[str, Any]) -> tuple[Any, Any, str | None, str | None]:
    """Return display value, raw value, display unit, raw unit."""
    for key in ("valueQuantity", "valueInteger", "valueDecimal", "valueString", "valueBoolean", "valueCodeableConcept", "valueDateTime", "valueRange", "valueRatio", "valueSampledData", "valueTime"):
        if key in r:
            raw = r[key]
            if key == "valueQuantity" and isinstance(raw, dict):
                unit = raw.get("unit") or raw.get("code")
                return raw.get("value"), raw.get("value"), str(unit) if unit is not None else None, str(raw.get("code")) if raw.get("code") is not None else None
            if key == "valueCodeableConcept":
                return _coding(raw), raw, None, None
            return raw, raw, None, None
    # A resource without a value still has a meaningful coded label.
    for key in ("code", "medicationCodeableConcept", "vaccineCode", "type"):
        if key in r:
            return _coding(r[key]) or r[key], r[key], None, None
    return {"resource_id": r.get("id")}, None, None, None


def _decode_notes(resources: tuple[RawResource, ...]) -> list[tuple[str, RawResource]]:
    notes: list[tuple[str, RawResource]] = []
    for raw in resources:
        r = raw.data
        forms = []
        if raw.resource_type == "DocumentReference":
            forms = [c.get("attachment", {}) for c in r.get("content", []) if isinstance(c, dict)]
        elif raw.resource_type == "DiagnosticReport":
            forms = r.get("presentedForm", []) or []
        for form in forms:
            value = form.get("data") if isinstance(form, dict) else None
            if not value:
                continue
            try:
                decoded = base64.b64decode(value, validate=True).decode("utf-8", errors="replace")
            except (ValueError, TypeError):
                decoded = str(value)
            if decoded.strip():
                notes.append((decoded.strip(), raw))
    return notes


def normalize(bundle: Bundle) -> PatientRecord:
    """Extract source-linked facts without discarding raw resource mappings."""
    patient = next((x for x in bundle.resources if x.resource_type == "Patient"), None)
    patient_id = patient.resource_id if patient else bundle.bundle_id
    facts: list[Fact] = []
    notes_by_text: dict[str, list[RawResource]] = {}
    decoded_notes = _decode_notes(bundle.resources)
    note_resource_ids = {raw.resource_id for _, raw in decoded_notes}
    for text, raw in decoded_notes:
        notes_by_text.setdefault(text, []).append(raw)

    for raw in bundle.resources:
        r, typ = raw.data, raw.resource_type
        # Notes are emitted once per identical payload below, keeping all refs.
        if typ in ("DocumentReference", "DiagnosticReport") and raw.resource_id in note_resource_ids:
            continue
        status = r.get("status")
        if typ == "Condition":
            status = "; ".join(f"{k}={v}" for k, v in (("clinical", _coding(r.get("clinicalStatus"))), ("verification", _coding(r.get("verificationStatus")))) if v) or status
        if typ == "AllergyIntolerance":
            status = "; ".join(f"{k}={v}" for k, v in (("clinical", _coding(r.get("clinicalStatus"))), ("verification", _coding(r.get("verificationStatus")))) if v) or status
        value, raw_value, unit, raw_unit = _value(typ, r)
        codes = _coding(r.get("code")) or _coding(r.get("medicationCodeableConcept")) or _coding(r.get("vaccineCode"))
        if typ == "Patient":
            value = {k: r.get(k) for k in ("name", "gender", "birthDate", "address", "deceasedBoolean", "deceasedDateTime") if r.get(k) is not None}
            raw_value = value
        src = SourceRef(bundle.bundle_id, typ, raw.resource_id)
        fact_id = f"{typ}/{raw.resource_id}"
        kind = "billing" if typ in {"Claim", "ExplanationOfBenefit"} else typ.lower()
        facts.append(Fact(fact_id, kind, value, str(status) if status is not None else None,
                          _dates(typ, r), codes, raw_value, unit, raw_unit, (src,),
                          {"resource_id": raw.resource_id, "references": _references(r)}))

    for text, sources in notes_by_text.items():
        refs = tuple(SourceRef(bundle.bundle_id, src.resource_type, src.resource_id, text=text) for src in sources)
        # Stable identity independent of Python hash randomization.
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        facts.append(Fact(f"note/{digest}", "note", text,
                          "; ".join(sorted({str(s.data.get('status')) for s in sources if s.data.get('status')})) or None,
                          tuple(d for src in sources for d in _dates(src.resource_type, src.data)),
                          (), text, None, None, refs, {"duplicate_source_count": len(refs)}))

    if not any(r.resource_type == "AllergyIntolerance" for r in bundle.resources):
        facts.append(Fact("allergy/not-recorded", "allergy_status", "No AllergyIntolerance resource is recorded in this bundle.",
                          "not-recorded", (), (), None, None, None,
                          (SourceRef(bundle.bundle_id, "Bundle", bundle.bundle_id),), {"meaning": "not_recorded"}))
    facts.sort(key=lambda f: (f.kind, f.dates[0].value if f.dates else "", f.fact_id))
    return PatientRecord(patient_id, tuple(facts), bundle.resources, bundle.bundle_id)


def _references(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "reference" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_references(v))
    elif isinstance(value, list):
        for item in value:
            found.extend(_references(item))
    return tuple(found)


def _priority(f: Fact) -> tuple[int, tuple[int, ...], str]:
    priorities = {"patient": 0, "allergyintolerance": 1, "allergy_status": 1,
                  "condition": 2, "medicationrequest": 3, "observation": 4,
                  "diagnosticreport": 5, "encounter": 6, "procedure": 7,
                  "immunization": 8, "medicationadministration": 9, "note": 10,
                  "billing": 99}
    # ISO-like dates sort lexically; invert character weights so the newest
    # item in each clinical class survives a tight budget before older items.
    date_value = f.dates[0].value if f.dates else ""
    newest_first = tuple(-ord(char) for char in date_value)
    return priorities.get(f.kind, 11), newest_first, f.fact_id


def _render_fact(f: Fact) -> str:
    return json.dumps({"kind": f.kind, "value": f.value, "status": f.status,
                       "dates": [{"kind": d.kind, "value": d.value} for d in f.dates],
                       "codes": f.codes, "raw_value": f.raw_value, "units": f.units,
                       "raw_unit": f.raw_unit,
                       "sources": [{"bundle_id": s.bundle_id, "resource_type": s.resource_type,
                                    "resource_id": s.resource_id} for s in f.sources]},
                      ensure_ascii=False, separators=(",", ":"), default=str)


def build_context(record: PatientRecord, budget: int = 900_000,
                  counter: Callable[[str], int] | None = None) -> ContextArtifact:
    """Select facts deterministically, counting with a caller-supplied counter.

    Without a counter, UTF-8 byte length / 4 is an explicitly labeled estimate.
    A fact is never split; facts that do not fit are reported as omitted.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    ordered = sorted(record.facts, key=_priority)
    candidates = [f for f in ordered if f.kind != "billing"]
    # Billing records remain normalized and retrievable, but are excluded from
    # the default narrative context by the initial selection policy.
    omitted: list[Fact] = [f for f in ordered if f.kind == "billing"]
    method = "provided_counter"
    def render(items: list[Fact]) -> str:
        header = json.dumps({"patient_id": record.patient_id, "bundle_id": record.bundle_id}, ensure_ascii=False, separators=(",", ":"))
        return header + "\n" + "\n".join(_render_fact(f) for f in items)
    count = counter or (lambda text: max(1, (len(text.encode("utf-8")) + 3) // 4))
    if counter is None:
        method = "estimated_utf8_bytes_div_4"
    # Selection follows a stable priority order, so fit can be found with a
    # logarithmic number of exact counts instead of one remote token-counting
    # request per resource. All later facts are lower-priority by policy.
    full_count = count(render(candidates))
    if full_count <= budget:
        fit = len(candidates)
    else:
        low, high = 0, len(candidates)
        while low < high:
            middle = (low + high + 1) // 2
            if count(render(candidates[:middle])) <= budget:
                low = middle
            else:
                high = middle - 1
        fit = low
    selected = candidates[:fit]
    omitted.extend(candidates[fit:])
    text = render(selected)
    token_count = count(text)
    # Verify the exact final rendering. Tokenizer boundary behavior can differ
    # slightly from a prefix's count; back off until the returned artifact
    # itself satisfies the stated budget.
    while selected and token_count > budget:
        omitted.insert(0, selected.pop())
        text = render(selected)
        token_count = count(text)
    return ContextArtifact(text, tuple(f.fact_id for f in selected), tuple(f.fact_id for f in omitted),
                           token_count, method, budget, token_count <= budget)

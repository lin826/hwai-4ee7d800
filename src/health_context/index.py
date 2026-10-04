"""Patient-scoped concept index: the retrieval layer for one FHIR bundle.

Every clinical resource is grouped under the concept it records (a LOINC lab,
a SNOMED condition, an RxNorm drug, ...). A concept keeps its full dated
timeline, so "most recent", "first", "how many" and "trend" are answered by
reading one item instead of hoping the right readings land in a top-k.

Clinical notes in this corpus are generated from the same structured data, so
they are split by section, de-duplicated, and searched as supporting text.
Every event and note chunk keeps the FHIR resource ids it came from.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .models import Bundle

# Kind words are appended to each concept's searchable text so questions such
# as "when was X diagnosed" or "which vaccines" favour the right resource type.
KIND_WORDS = {
    "lab": "lab test result measurement observation",
    "condition": "condition diagnosis diagnosed problem disorder",
    "medication": "medication medicine drug prescribed prescription",
    "procedure": "procedure performed",
    "immunization": "immunization vaccine vaccination shot",
    "encounter": "encounter visit appointment",
    "allergy": "allergy allergic allergies intolerance reaction",
    "careplan": "care plan",
    "device": "device implant",
    "imaging": "imaging study scan",
}

# Small, auditable synonym table for labs: lay or abbreviated terms keyed by a
# phrase that appears in the coded display name. Extend it only with eval evidence.
ALIASES = {
    "hemoglobin a1c": "hba1c a1c glycated blood sugar glycemic diabetes",
    "glucose": "blood sugar glycemic",
    "blood pressure": "bp hypertension",
    "systolic": "bp",
    "diastolic": "bp",
    "body mass index": "bmi obesity",
    "body weight": "weight",
    "body height": "height",
    "heart rate": "pulse",
    "cholesterol": "lipid lipids",
    "low density lipoprotein": "ldl lipid",
    "high density lipoprotein": "hdl lipid",
    "triglycerides": "lipid",
    "creatinine": "kidney renal",
    "glomerular filtration": "egfr kidney renal",
    "urea nitrogen": "bun kidney renal",
    "hemoglobin [mass/volume]": "hgb anemia",
    "tobacco smoking status": "smoker smoking",
    "pain severity": "pain",
}

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "most",
        "of",
        "on",
        "or",
        "recent",
        "latest",
        "last",
        "first",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "whom",
        "with",
        "ever",
        "any",
        "patient",
        "patients",
        "value",
        "values",
    ]
)

_TOKEN = re.compile(r"[a-z0-9]+")
# Notes here are generated from the structured records, so a note section is
# supporting evidence: it ranks below a structured concept with a similar score.
NOTE_WEIGHT = 0.5
# Bonus when every token of a concept's name appears in the query, so
# "dental care" prefers "Dental care" over "Patient referral for dental care".
NAME_MATCH_BONUS = 2.0
# Word forms the plural folding below cannot align.
_LEMMAS = {
    "allergic": "allergy",
    "diabetic": "diabetes",
    "vaccinated": "vaccine",
    "smoke": "smoking",
    "smoker": "smoking",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(text.casefold()):
        if token in STOPWORDS:
            continue
        token = _LEMMAS.get(token, token)
        # Minimal plural folding keeps "allergies"/"allergy" and "readings"/"reading" aligned.
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


# SNOMED semantic tags such as "(disorder)" are not part of a concept's name.
_TAG = re.compile(r"\s*\([a-z /]+\)\s*$")


@dataclass(frozen=True)
class Event:
    date: str | None
    resource_id: str
    resource_type: str
    value: float | str | None = None
    unit: str | None = None
    status: str | None = None
    end: str | None = None
    encounter_id: str | None = None
    detail: str | None = None


@dataclass
class Concept:
    concept_id: str
    kind: str
    display: str
    system: str | None
    code: str | None
    category: str | None = None
    events: list[Event] = field(default_factory=list)

    @property
    def latest(self) -> Event | None:
        dated = [e for e in self.events if e.date]
        return dated[-1] if dated else (self.events[-1] if self.events else None)

    @property
    def first(self) -> Event | None:
        dated = [e for e in self.events if e.date]
        return dated[0] if dated else (self.events[0] if self.events else None)

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(e.resource_id for e in self.events)

    def search_text(self) -> str:
        display = self.display.casefold()
        aliases = (
            " ".join(v for k, v in ALIASES.items() if k in display)
            if self.kind == "lab"
            else ""
        )
        details = " ".join(sorted({e.detail for e in self.events if e.detail}))
        return " ".join(
            filter(
                None,
                [
                    self.display,
                    self.category,
                    self.code,
                    aliases,
                    details,
                    KIND_WORDS.get(self.kind, self.kind),
                ],
            )
        )


@dataclass
class NoteChunk:
    chunk_id: str
    section: str
    text: str
    # One entry per note that contains this exact section text:
    # (visit key, note date, source resource ids).
    occurrences: list[tuple[str | None, str | None, tuple[str, ...]]] = field(
        default_factory=list
    )

    @property
    def resource_ids(self) -> tuple[str, ...]:
        return tuple(rid for _, _, ids in self.occurrences for rid in ids)


@dataclass(frozen=True)
class Hit:
    item: Concept | NoteChunk
    score: float

    @property
    def item_id(self) -> str:
        return (
            self.item.concept_id
            if isinstance(self.item, Concept)
            else self.item.chunk_id
        )


def _concept_code(concept: Any) -> tuple[str | None, str | None, str]:
    """Return (system, code, display) for the first coding of a CodeableConcept."""
    if not isinstance(concept, dict):
        return None, None, "unknown"
    coding = next((c for c in concept.get("coding") or [] if isinstance(c, dict)), {})
    display = (
        concept.get("text") or coding.get("display") or coding.get("code") or "unknown"
    )
    return coding.get("system"), coding.get("code"), str(display)


def _ref_id(ref: Any) -> str | None:
    value = ref.get("reference") if isinstance(ref, dict) else None
    if isinstance(value, str) and value.startswith("urn:uuid:"):
        return value.removeprefix("urn:uuid:")
    return value


def _status(r: dict[str, Any]) -> str | None:
    clinical = r.get("clinicalStatus")
    if isinstance(clinical, dict):
        return _concept_code(clinical)[1]
    return r.get("status")


class PatientIndex:
    """All of one patient's retrievable material, grouped by clinical concept."""

    def __init__(self, bundle: Bundle):
        patient = next(
            (r.data for r in bundle.resources if r.resource_type == "Patient"), {}
        )
        self.concepts: dict[str, Concept] = {}
        self.notes: dict[str, NoteChunk] = {}
        # Some prescriptions reference a Medication resource instead of
        # carrying the drug code inline.
        self._medications = {
            r.resource_id: r.data.get("code")
            for r in bundle.resources
            if r.resource_type == "Medication"
        }
        for raw in bundle.resources:
            self._add(raw.resource_type, raw.data)
        self._finish(
            patient,
            bundle.bundle_id,
            {r.resource_id: r.resource_type for r in bundle.resources},
        )

    @classmethod
    def from_parts(
        cls,
        patient: dict[str, Any],
        bundle_id: str,
        concepts: dict[str, Concept],
        notes: dict[str, NoteChunk],
        resource_types: dict[str, str],
    ) -> PatientIndex:
        """Rebuild an index from stored rows (see ``store.py``) without the bundle."""
        index = cls.__new__(cls)
        index.concepts, index.notes = concepts, notes
        index._finish(patient, bundle_id, resource_types)
        return index

    def _finish(
        self, patient: dict[str, Any], bundle_id: str, resource_types: dict[str, str]
    ) -> None:
        self.patient = patient
        self.bundle_id = bundle_id
        self.patient_id: str = patient.get("id", bundle_id)
        self.resource_types = resource_types
        self.counts = Counter(resource_types.values())
        self.by_encounter: dict[str, list[str]] = defaultdict(list)
        for concept in self.concepts.values():
            concept.events.sort(key=lambda e: (e.date or "", e.resource_id))
            for e in concept.events:
                if e.encounter_id:
                    self.by_encounter[e.encounter_id].append(e.resource_id)
        self._build_search()

    # -- construction -------------------------------------------------------

    def _concept(self, kind: str, concept: Any, category: str | None = None) -> Concept:
        system, code, display = _concept_code(concept)
        key = f"{kind}:{code or display}"
        if key not in self.concepts:
            self.concepts[key] = Concept(key, kind, display, system, code, category)
        return self.concepts[key]

    def _add(self, typ: str, r: dict[str, Any]) -> None:
        rid = str(r.get("id", ""))
        encounter = _ref_id(r.get("encounter"))
        if typ == "Observation":
            self._add_observation(r, rid, encounter)
        elif typ == "Condition":
            self._concept("condition", r.get("code")).events.append(
                Event(
                    r.get("onsetDateTime") or r.get("recordedDate"),
                    rid,
                    typ,
                    status=_status(r),
                    end=r.get("abatementDateTime"),
                    encounter_id=encounter,
                )
            )
        elif typ == "MedicationRequest":
            drug = r.get("medicationCodeableConcept") or self._medications.get(
                _ref_id(r.get("medicationReference")) or ""
            )
            self._concept("medication", drug).events.append(
                Event(
                    r.get("authoredOn"),
                    rid,
                    typ,
                    status=_status(r),
                    encounter_id=encounter,
                )
            )
        elif typ == "Procedure":
            period = r.get("performedPeriod") or {}
            self._concept("procedure", r.get("code")).events.append(
                Event(
                    r.get("performedDateTime") or period.get("start"),
                    rid,
                    typ,
                    status=_status(r),
                    end=period.get("end"),
                    encounter_id=encounter,
                )
            )
        elif typ == "Immunization":
            self._concept("immunization", r.get("vaccineCode")).events.append(
                Event(
                    r.get("occurrenceDateTime"),
                    rid,
                    typ,
                    status=_status(r),
                    encounter_id=encounter,
                )
            )
        elif typ == "Encounter":
            period = r.get("period") or {}
            reasons = "; ".join(_concept_code(c)[2] for c in r.get("reasonCode") or [])
            concept = (r.get("type") or [None])[0]
            self._concept("encounter", concept).events.append(
                Event(
                    period.get("start"),
                    rid,
                    typ,
                    status=_status(r),
                    end=period.get("end"),
                    detail=f"reason: {reasons}" if reasons else None,
                )
            )
        elif typ == "AllergyIntolerance":
            reactions = "; ".join(
                _concept_code(m)[2]
                for x in r.get("reaction") or []
                for m in x.get("manifestation") or []
            )
            self._concept("allergy", r.get("code")).events.append(
                Event(
                    r.get("recordedDate") or r.get("onsetDateTime"),
                    rid,
                    typ,
                    status=_status(r),
                    detail=f"reaction: {reactions}" if reactions else None,
                )
            )
        elif typ == "CarePlan":
            period = r.get("period") or {}
            self._concept("careplan", (r.get("category") or [None])[-1]).events.append(
                Event(
                    period.get("start"),
                    rid,
                    typ,
                    status=_status(r),
                    end=period.get("end"),
                    encounter_id=encounter,
                )
            )
        elif typ == "Device":
            self._concept("device", r.get("type")).events.append(
                Event(r.get("manufactureDate"), rid, typ, status=_status(r))
            )
        elif typ == "ImagingStudy":
            series = (r.get("series") or [{}])[0]
            detail = " ".join(
                filter(
                    None,
                    [
                        (series.get("bodySite") or {}).get("display"),
                        (series.get("modality") or {}).get("display"),
                    ],
                )
            )
            self._concept(
                "imaging", (r.get("procedureCode") or [None])[0]
            ).events.append(
                Event(
                    r.get("started"),
                    rid,
                    typ,
                    status=_status(r),
                    encounter_id=encounter,
                    detail=detail or None,
                )
            )
        elif typ in ("DocumentReference", "DiagnosticReport"):
            self._add_note(typ, r, rid, encounter)
        # Claim/ExplanationOfBenefit, Provenance, SupplyDelivery and
        # MedicationAdministration are intentionally not indexed:
        # billing and supply logistics are outside the clinical question scope.

    def _add_observation(
        self, r: dict[str, Any], rid: str, encounter: str | None
    ) -> None:
        category = _concept_code((r.get("category") or [None])[0])[1]
        date = r.get("effectiveDateTime") or (r.get("effectivePeriod") or {}).get(
            "start"
        )
        # Panels such as blood pressure carry their values in components; each
        # component is its own concept so systolic and diastolic stay separate.
        parts = r.get("component") if "component" in r else [r]
        for part in parts:
            value, unit = _observation_value(part)
            if value is None and part is not r:
                continue
            self._concept("lab", part.get("code"), category).events.append(
                Event(
                    date,
                    rid,
                    "Observation",
                    value=value,
                    unit=unit,
                    status=r.get("status"),
                    encounter_id=encounter,
                )
            )

    def _add_note(
        self, typ: str, r: dict[str, Any], rid: str, encounter: str | None
    ) -> None:
        forms = (
            [c.get("attachment", {}) for c in r.get("content", [])]
            if typ == "DocumentReference"
            else r.get("presentedForm") or []
        )
        date = r.get("date") or r.get("effectiveDateTime")
        if encounter is None:
            encounter = _ref_id(
                next(iter((r.get("context") or {}).get("encounter") or []), None)
            )
        visit = encounter or date
        for form in forms:
            data = form.get("data") if isinstance(form, dict) else None
            if not data:
                continue
            text = base64.b64decode(data).decode("utf-8", errors="replace")
            for section, body in _sections(text):
                chunk_id = (
                    "note:"
                    + hashlib.sha256(f"{section}\n{body}".encode()).hexdigest()[:12]
                )
                chunk = self.notes.setdefault(
                    chunk_id, NoteChunk(chunk_id, section, body)
                )
                # DocumentReference and DiagnosticReport carry the same payload
                # for one encounter; merge them into a single occurrence.
                for i, (v, d, ids) in enumerate(chunk.occurrences):
                    if v == visit and rid not in ids:
                        chunk.occurrences[i] = (v, d, ids + (rid,))
                        break
                else:
                    chunk.occurrences.append((visit, date, (rid,)))

    # -- search -------------------------------------------------------------

    def _build_search(self) -> None:
        items: list[Concept | NoteChunk] = [
            *self.concepts.values(),
            *self.notes.values(),
        ]
        self._items = items
        self._docs = [
            Counter(
                tokenize(
                    i.search_text()
                    if isinstance(i, Concept)
                    else f"{i.section} {i.text}"
                )
            )
            for i in items
        ]
        self._names = [
            set(tokenize(_TAG.sub("", i.display))) if isinstance(i, Concept) else set()
            for i in items
        ]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 1.0
        )
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(items)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(
        self,
        query: str,
        k: int = 5,
        kinds: set[str] | None = None,
        include_notes: bool = True,
    ) -> list[Hit]:
        """BM25 over concepts and note sections, scoped to this patient."""
        terms = tokenize(query)
        query_terms = set(terms)
        k1, b = 1.2, 0.75
        hits = []
        for item, doc, length, name in zip(
            self._items, self._docs, self._lengths, self._names
        ):
            if isinstance(item, NoteChunk):
                if not include_notes or (kinds and "note" not in kinds):
                    continue
            elif kinds and item.kind not in kinds:
                continue
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    score += (
                        self._idf[t]
                        * tf
                        * (k1 + 1)
                        / (tf + k1 * (1 - b + b * length / self._avg_len))
                    )
            if score <= 0:
                continue
            if isinstance(item, NoteChunk):
                score *= NOTE_WEIGHT
            elif name and name <= query_terms:
                score += NAME_MATCH_BONUS
            hits.append(Hit(item, score))
        hits.sort(key=lambda h: (-h.score, h.item_id))
        return hits[:k]

    def concepts_of(self, kind: str) -> list[Concept]:
        return sorted(
            (c for c in self.concepts.values() if c.kind == kind),
            key=lambda c: c.display.casefold(),
        )


def _observation_value(part: dict[str, Any]) -> tuple[float | str | None, str | None]:
    if isinstance(part.get("valueQuantity"), dict):
        q = part["valueQuantity"]
        return q.get("value"), q.get("unit") or q.get("code")
    if "valueCodeableConcept" in part:
        return _concept_code(part["valueCodeableConcept"])[2], None
    for key in ("valueString", "valueInteger", "valueBoolean", "valueDateTime"):
        if key in part:
            return part[key], None
    return None, None


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown note into (heading, body) pairs, dropping empty bodies."""
    sections, heading, lines = [], "preamble", []
    for line in text.splitlines():
        if line.startswith("#"):
            if any(x.strip() for x in lines):
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("#").strip().casefold(), []
        else:
            lines.append(line)
    if any(x.strip() for x in lines):
        sections.append((heading, "\n".join(lines).strip()))
    return sections

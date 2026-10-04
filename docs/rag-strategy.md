# Retrieval Strategy

## Problem

The README asks for retrieval over whatever does not fit in the compact context. The questions it uses as examples are about one exact value and its date, a trend across readings, and an absence. Those are structured questions. Splitting the bundle into text chunks and ranking them by embedding similarity answers them poorly: top-k chunks cannot guarantee that the *most recent* of eight readings is among them, and a trend needs every reading, not the k most similar ones.

## What the data says

Measured over the 109 bundles in `data/`:

- **Small coded vocabulary.** There are 217 distinct Observation codes in the whole corpus and at most 92 for one patient. Everything clinical carries a LOINC, SNOMED, RxNorm or CVX code.
- **Notes are derived text.** Each encounter note restates the patient's conditions, medications and procedures that already exist as structured resources, and the same payload appears in both a DocumentReference and a DiagnosticReport.
- **Repetition, not variety, drives size.** The largest patient has 9,610 Observations but only 104 lab concepts after splitting blood-pressure components.

## Design

```text
FHIR bundle ──> PatientIndex (index.py)
                 ├─ concepts: one per coded thing, with its full dated timeline
                 │   lab:4548-4  Hemoglobin A1c   8 readings, each with value/unit/date/resource id
                 │   condition:714628002  Prediabetes   onset, status, abatement
                 ├─ note sections: split by heading, de-duplicated, every source id kept
                 └─ BM25 over concept names + aliases + kind words, notes down-weighted

Prompt  = catalog (tools.catalog): every concept on one line with count, span, latest value
Tools   = search_records(query) / get_timeline(concept_id, since, until) / get_encounter(id)
```

1. **Group by concept, not by chunk.** Each item in the index is a clinical concept with its whole timeline. "Most recent", "first", "how many" and "trend" are read off one item deterministically.
2. **The catalog is the always-on context.** It tells the model what exists, including an explicit "no AllergyIntolerance recorded (not confirmed absent)" line. Its size across the corpus is about 0.8k to 7.8k tokens by a bytes/4 estimate (median 3.8k). This is not an authoritative Anthropic count.
3. **Tools return detail with citations.** Every timeline row carries its `ResourceType/id`. Long flat series are collapsed into runs, for example `2019-02-20..2026-08-12 2.43 % x144`, so the largest timelines stay readable.
4. **Lexical ranking, by choice.** BM25 over concept names, with a small lab alias table (`hba1c`, `bp`, `ldl`, ...), kind words ("diagnosed" favours conditions), a name-match bonus, and note sections weighted at 0.5 because they repeat structured data. Every rule is visible in `index.py` and can be explained line by line.
5. **The model bridges vocabulary.** Lay terms that keyword search cannot match ("statin" vs "simvastatin", "high blood pressure" vs "hypertension") are the model's job: it reads the catalog and calls tools with the coded name. Embeddings are deferred until an end-to-end eval shows this fails.

## Eval

`uv run python -m health_context rag-eval` generates questions with known answers from every bundle. Ground truth is read from the raw JSON by separate code in `rag_eval.py` that shares nothing with the index. A case passes *recall@k* when the resource holding the answer is in one of the top-k items, and passes *answer@1* when the top item's own first or latest event reproduces the ground-truth resource, date and value.

Current result, 109 patients, 3 questions per category per patient, 4 seconds:

| category | n | recall@1 | recall@5 | answer@1 |
|---|---:|---:|---:|---:|
| lab latest value | 327 | 1.000 | 1.000 | 1.000 |
| condition first diagnosed | 324 | 1.000 | 1.000 | 1.000 |
| medication last prescribed | 286 | 1.000 | 1.000 | 1.000 |
| immunization latest | 315 | 1.000 | 1.000 | 1.000 |
| procedure latest | 322 | 1.000 | 1.000 | 1.000 |
| hand-written paraphrases | 3 | 0.667 | 1.000 | 0.667 |
| **all** | 1577 | 0.999 | 1.000 | 0.999 |

What the eval drove: the first run scored medication recall@1 0.535 because note "plan" sections repeating the drug name outranked the prescription itself, and "dental care" tied with "Patient referral for dental care". Down-weighting notes and adding the name-match bonus fixed both without regressing other categories.

## Limits

- **Templated questions are easy for lexical search.** They use the coded display names, so these numbers show the index finds a named concept and reads its timeline correctly. They do not measure lay-language questions. The remaining miss ("Is her blood sugar getting worse?") ranks Glucose above HbA1c, which is defensible but not the hand-labelled answer.
- **Known vocabulary gaps** for direct search: drug classes ("statin", "blood thinner") and lay condition names ("high blood pressure"). These need the model-in-the-loop eval below before deciding on aliases or embeddings.
- **Not yet indexed:** billing, SupplyDelivery, MedicationAdministration, Provenance. Shared practitioner and organization files are not joined.
- Ground truth and index both break same-date ties by resource id.

## Next

1. Agent loop: catalog in the prompt plus the three tools via Anthropic tool use, recording which tools and ids the model used. Requires `ANTHROPIC_API_KEY`.
2. End-to-end eval on the same generated questions plus a lay-language set, graded on value, date and cited resource id, with absence questions that check the model does not invent answers.
3. Count the catalog with `count_tokens` on `claude-opus-5` and compare catalog plus tools against the full compact context on the same eval.
4. Replace the older fact-level `retrieval.py` once the agent loop uses the index.

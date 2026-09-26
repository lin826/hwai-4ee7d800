# Initial Architecture Boundaries

## Goal and non-goals

The system builds a concise patient context from one synthetic Synthea FHIR Bundle, gives a question-answering flow access to omitted material, and evaluates whether important facts remain accurate and traceable. The 900,000-token ceiling is a hard cap for the exact context message, counted with Anthropic's endpoint when credentials are available.

This is an engineering exercise on synthetic data, not clinical decision support or clinical validation. It should not present “no record found” as proof that a condition is absent.

## Data flow

```text
FHIR Bundle (+ shared practitioner / organization files)
                  |
                  v
             Ingestion
                  |
                  v
       Normalized facts + provenance
            /                  \
           v                    v
 Compact context selector    Retrieval index
           |                    |
           v                    v
 Context text + budget      Evidence records
            \                  /
             v                v
              Evaluation / demo QA
```

## Component responsibilities

- **Ingestion** reads bundle entries, retains each resource's type, id, raw content, and references, and reports malformed or dangling references. It does not decide clinical importance.
- **Normalization** extracts candidate facts and standardizes representation where safe. It keeps the original code/value/unit and resource status alongside normalized forms. Dates remain typed by meaning rather than collapsed into one generic timestamp.
- **Context selection and rendering** applies explicit prioritization and size policy. It must retain provenance, represent unknown/not-recorded states carefully, and report what was excluded. Deterministic output is the default; any generated note summary must be separately identifiable and source-linked.
- **Retrieval** indexes material that is not fully represented in the compact context. Queries must be patient-scoped and results include source resource identifiers, dates, and relevant evidence. Begin with deterministic filters/text search; embeddings are optional pending eval evidence.
- **Evaluation** measures exact fact/date/unit retrieval, trend handling, absence/conflict semantics, retrieval evidence quality, and token-budget compliance. Scores should distinguish answer correctness from evidence correctness.

## Cross-cutting invariants

1. Every surfaced fact can be traced to a source resource (and source span for note-derived claims where feasible).
2. A missing record is rendered as “not recorded” or equivalent, not as a definitive negative.
3. Conflicting records remain visible as conflicts unless an explicit, auditable policy resolves them.
4. Numeric clinical values retain units, dates, and source status; conversions do not erase raw values.
5. Context selection and retrieval share the same normalized facts/provenance model so omitted facts remain retrievable.
6. Output records the token count and counting method; estimates cannot be reported as authoritative compliance.

## Shared interfaces

The data inventory is documented in `docs/data-investigation.md`. The initial contracts are:

- `load_bundle(path) -> Bundle`: `Bundle` holds patient bundle identity and ordered raw resources; each raw resource carries `resource_type`, `resource_id`, and the original mapping.
- `normalize(bundle) -> PatientRecord`: `PatientRecord` contains `patient_id`, `facts`, and `raw_resources`. A `Fact` has `kind`, `value`, `status`, typed `dates`, `codes`, `units`, and `sources`. `SourceRef` carries `resource_type`, `resource_id`, optional source text/span, and bundle identity. Raw value/code/unit must remain available when normalized forms are added.
- `build_context(record, budget, counter) -> ContextArtifact`: artifact has exact rendered `text`, selected fact/source IDs, omitted fact/source IDs, `token_count`, `count_method`, and `budget_satisfied`. If no authoritative counter is configured, the count is marked estimated and cannot claim the challenge cap is verified.
- `retrieve(record, query, limit=10) -> list[Evidence]`: each evidence item carries text or fact, source refs, relevant typed dates, and rank/match metadata. Patient scope is mandatory.
- `evaluate(record, context, retriever, cases, answers=None) -> EvaluationReport`: cases define question, expected answer, optional `coverage_expected` literals for provider-free coverage scoring, required evidence, and whether retrieval is allowed. Report answer/fact accuracy, evidence accuracy, and budget compliance separately.

### Selection policy v0

Include identity; allergies as recorded/not recorded with source and status; active and historical Conditions with status/onset/abatement; medications with status and dates; high-signal observations with codes, raw values, units and dates; and concise encounter/procedure timeline. Do not include most billing resources in default context. Deduplicate note text only when payloads match and preserve both source links. Retrieval should cover every normalized fact, including facts not selected into context. This policy is an engineering baseline for eval-driven revision, not a claim of clinical completeness.

Detailed field-level Python types should mirror these contracts and remain intentionally small. Any proposal to change absence/conflict semantics or protected facts is a lead-owned design decision.

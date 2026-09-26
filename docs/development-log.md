# Development Log

This log records the project decisions and iteration sequence so the design can be explained and revised. The work is an engineering prototype over synthetic Synthea data, not a clinically validated system.

## Iteration 1: Understand the brief and inspect the repository

**Question:** What needs to be built, and what constraints are fixed?

**Findings:** The repository initially contained the challenge README, dataset documentation, and a compressed FHIR archive, but no implementation. The challenge requires a compact patient context with a 900,000-token ceiling, an evaluation suite, and retrieval for omitted information.

**Decision:** Keep the first implementation deterministic and provider-independent. Treat Anthropic's token counting endpoint as the authority for budget compliance; any local estimate is diagnostic only. Keep synthetic-data results distinct from clinical validation claims.

## Iteration 2: Inventory the real bundle patterns

**Question:** Which resource patterns and scale constraints should shape the design?

**Method:** Extracted the archive into a temporary directory and inspected JSON using the Python standard library. The source archive was not modified. See [data-investigation.md](data-investigation.md) for counts, representative bundles, and source identifiers.

**Findings affecting design:**

- The corpus contains 109 patient bundles, with patient bundle sizes from about 108 KB to 52 MB.
- Observations dominate resource count; Claim and ExplanationOfBenefit are numerous paired billing records.
- Each encounter's note payload appears in both DocumentReference and DiagnosticReport representations. Matching payloads can be stored once for context/retrieval while retaining both source links.
- Resource statuses and date field meanings vary and carry useful semantics.
- Missing AllergyIntolerance resources are common and cannot be interpreted as proof of no allergies.
- In-bundle UUID references were intact in the inspected corpus; shared practitioner and organization references use identifier queries.

**Decision:** Preserve source IDs, resource status, raw values/units, and semantically named dates. Represent absent allergy data as “not recorded.” Keep billing out of the default compact context while leaving it available to retrieval. Deduplicate note text only when payloads match.

## Iteration 3: Define architecture boundaries and shared contracts

**Question:** How can components be developed independently without each inventing a schema?

**Decision:** Separate ingestion, normalization, context selection/rendering, retrieval, and evaluation. Establish plain Python dataclasses in `src/health_context/models.py` as the shared contract. Keep clinical selection policy out of parsing and keep all surfaced facts source-linked.

**Important contract choices:**

- Context artifacts report selected and omitted fact IDs, the count method, and whether the supplied budget is satisfied.
- Evidence carries typed dates and source references.
- Evaluation reports answer/fact coverage, evidence correctness, and budget compliance separately.
- Provider-free coverage is explicitly a proxy; only supplied answer strings are scored as answers.

## Iteration 4: Delegate bounded implementation

**Work split:** Implement the pipeline, retrieval, and evaluation modules against the shared contracts, each in a separate source file. The lead retained ownership of shared schemas, selection policy, integration, and review.

**Implemented:**

- `pipeline.py` loads FHIR Bundles, extracts source-linked facts, preserves raw clinical fields, deduplicates matching note payloads, handles missing allergy records, and builds a deterministic budgeted context.
- `retrieval.py` provides patient-record-scoped lexical retrieval with source references and dates.
- `evaluation.py` defines hand-checkable challenge cases and provider-independent scoring helpers.

## Iteration 5: Integration review and current state

**Review changes:** Newer facts are ordered ahead of older facts within their priority class so tight budgets do not automatically favor stale observations. Context fit selection uses a logarithmic number of counts over a priority-ordered prefix, then verifies the final rendered text. A too-small budget that cannot contain even the header is represented by `budget_satisfied = false`.

**Not yet validated:** No tests, type checker, linter, end-to-end run, or Anthropic token-count request has been executed in this iteration. The estimate mode is not evidence that any context meets the challenge's authoritative token ceiling. The current retrieval is lexical and does not cover synonyms or semantic similarity. Note text is retained in full rather than summarized.

**Next validation sequence:**

1. Add a reproducible command-line demo that loads the challenge patient bundle, writes a context artifact, and displays retrieval/evaluation summaries.
2. Run the eval cases first with provider-free coverage scoring, then with reviewed answers and required source IDs.
3. Configure Anthropic counting and check budget behavior on both the sample patient and largest bundle.
4. Run repository formatting/type/lint checks and address integration defects.
5. Expand held-out and adversarial cases for conflicts, medication status, dates/units, retrieval-only facts, and explicit absence.

# Agent Guide

## Project goal

Build a demoable Python pipeline that converts one patient's synthetic Synthea FHIR Bundle into (1) a compact, source-grounded context under the challenge's 900,000-token limit, (2) retrieval over material left out of that context, and (3) evaluations of answer quality and evidence fidelity. The challenge specification is in `README.md`; dataset details are in `DATA.md`.

## Operating rules

- Inspect the relevant data and existing code before making design assumptions. The repository currently starts with the challenge and a compressed dataset, so do not assume an implementation or interface exists.
- Keep core ingestion, normalization, context assembly, and evals deterministic and runnable without model credentials. Add model calls only where they provide a demonstrated benefit, and keep their outputs traceable to source material.
- Preserve provenance for every emitted or retrieved fact: patient/bundle identity, FHIR resource type and id, and clinically meaningful date(s). Retain raw values and units when normalizing measurements.
- Distinguish “not recorded” from a confirmed negative. Preserve resource status and uncertainty; do not silently resolve contradictory records or infer causality.
- Keep compact-context selection policy separate from parsing and retrieval. Do not encode clinical prioritization as incidental parser behavior.
- Treat the 900,000-token cap as a hard product requirement. Use the specified Anthropic counting endpoint when available; label any alternate tokenizer as an estimate, never as authoritative compliance.
- This challenge data is synthetic. Do not claim clinical validity or readiness for real patient records.
- Keep changes focused, explain assumptions, and include reproducible commands and source references in investigation reports. Do not add dependencies without a clear need.

## Collaboration boundaries

- The lead owns clinical prioritization, unknown/conflict semantics, shared data contracts, and integration.
- Agents may investigate read-only or implement a bounded component after the shared interfaces are agreed.
- Do not independently redefine shared schemas, protected-fact rules, or eval acceptance criteria. Raise proposals with evidence for lead review.
- For parallel code work, use separate worktrees/branches where available; report changed files, validation performed, assumptions, and unresolved issues.

## Expected pipeline boundaries

1. **Ingestion:** load a FHIR Bundle and preserve resource identity and references.
2. **Normalization:** expose useful typed facts while retaining raw fields, statuses, dates, units, and provenance.
3. **Context selection/rendering:** select and render high-value facts under the budget; record exclusions and count method.
4. **Retrieval:** search non-context facts and return evidence with provenance.
5. **Evaluation:** assess exactness, temporal reasoning, missing/conflicting data behavior, evidence grounding, and budget compliance.

See `docs/architecture.md` for component boundaries, `docs/data-investigation.md` for corpus findings, and `docs/development-log.md` for the iteration history and open validation work. Shared interfaces are defined in `src/health_context/models.py`.

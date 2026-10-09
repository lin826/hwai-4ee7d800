# End-to-End Eval with LFM2.5-8B-A1B

## Setup

- Model: `hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0`, served locally by Ollama 0.32.5 (OpenAI-compatible API). Ollama returns structured `tool_calls` for this model, so the Pythonic-call fallback in `llm.py` is not exercised.
- Command: `uv run python -m health_context qa-eval --patients 10 --store store --json <file>`. The 10 patients are a deterministic sample that includes the README's sample patient. That gives 72 questions: one generated question per category per patient, the README sample questions, and two absence questions per patient.
- Grading is deterministic (`qa_eval.py`). An answer is correct when it states the ground-truth date (and value, for labs), or says "not recorded" for absence questions without stating a date. Citation means the answer contains the ground-truth FHIR resource id.
- Ground truth comes from the raw bundle JSON, computed independently of the index. Each failure below was checked against the index and the raw resources; none came from a wrong answer key.

## Results

| category | n | original prompt | tool-first prompt |
|---|---:|---:|---:|
| all | 72 | **0.847** | 0.889 |
| allergy absent | 9 | 1.000 | 1.000 |
| condition absent | 10 | **1.000** | 0.700 |
| condition first diagnosed | 10 | 1.000 | 0.800 |
| lab latest | 10 | 0.700 | 0.800 |
| immunization latest | 10 | 0.600 | 1.000 |
| procedure latest | 10 | 0.700 | 0.900 |
| medication latest | 9 | 1.000 | 1.000 |
| README sample questions | 3 | 0.667 | 1.000 |
| cited the gold resource id | | 0.000 | 0.019 |
| tool calls per question | | 0.19 | 0.60 |
| seconds per question | | 5.0 | 7.0 |

The "tool-first" prompt told the model to call `get_timeline` before stating any value or date and to cite the row's resource id.

## What the failures show

- **The model answers from the catalog instead of calling tools.** With the original prompt it made 0.19 tool calls per question. Every failure was a date read from the wrong catalog line or invented. Example: it reported DAST-10's date from the adjacent AUDIT-C line.
- **The stricter prompt trades errors for invented diagnoses.** It fixed most date errors but produced one fabricated diagnosis date for a condition the patient never had ("osteoporotic fracture ... 2016-12-09", no tool call). It also called `get_timeline` with unrelated or invented concept ids.
- **The model does not cite.** Even when it used tools, 22 of 23 answers omitted the resource id that the tool output shows on every row.
- **The grader has limits.** One tool-first failure was a correct absence answer ("records do not contain any documented diagnosis ...") whose wording missed the pattern and which mentioned the record's date span. The grader also folds typographic hyphens and spaces (`2025‑09‑29`, `6.31 %`) to ASCII, because the model emits them.

**Decision.** Keep the original prompt. Its 100% absence accuracy, with no invented diagnoses, matters more in a clinical setting than the higher date accuracy of the tool-first prompt.

## Next experiments

1. **Retrieve in code, then let the model read.** Run `search_records` on the question in code and put the top concept's timeline (with resource ids) in the prompt, instead of relying on a small model to choose tools.
2. **Check citations in code.** Reject or flag answers whose dates or ids do not appear in the retrieved rows.
3. **Make catalog lines unambiguous.** Immunization and procedure lines show only a date span; add an explicit "latest YYYY-MM-DD".
4. **Use a larger sample.** Ten patients give wide confidence intervals: one question moves a category by 0.1.

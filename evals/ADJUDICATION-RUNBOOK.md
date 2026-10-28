# Adjudication Runbook — legal_retrieval_v2

## Purpose

Independent human adjudication of disputed gold-row labels before release. Dual
engineer annotation and agreement statistics (`evals/reports/iaa_wave3.json`,
produced by `scripts/compute_iaa.py`) measure consistency but never settle a
legal-interpretation dispute. This runbook is the procedure for the human who does.

## Who adjudicates (manifest `sme_requirements`, verbatim)

- **Role**: practising lawyer.
- **Domain**: Indian contract law.
- **Independence**: independent of whoever produced the labels (not one of the
  wave-3 annotators, not their direct report, no shared incentive).
- **Commitment**: signs off on disputed cases.
- `named_adjudicator` is currently `null`; the manifest is updated with the real
  name/stable identifier when an SME is engaged. Rows needing sign-off: 41
  candidate rows with `provenance.adjudication.status=pending_sme`, including
  the 6 `retrieval_relevance` disagreements recorded in
  `evals/reports/iaa_wave3.json` (multi_document-lr2-003,
  no_evidence_vs_ingestion_failure-lr2-002/003, partial_coverage_answer-lr2-001/003,
  unanswerable-lr2-002).

## SME review checklist (all 7 items, recorded per row)

1. **Proposition-to-evidence support** — does every expected claim rest on the
   corrected/extracted text of the cited document (span-verbatim, page-accurate)?
2. **Near-miss legal meaning** — does the decoy clause differ in legal effect,
   not merely wording (e.g. LD rate vs cap, "shall" vs "may")?
3. **Agreement-vs-amendment conflicts** — where a corrigendum/addendum overlaps
   the base text, is the operative expectation the amended term?
4. **Jurisdiction / version / authority relevance** — is the expected answer
   grounded in the governing edition and jurisdiction, with the decoy document
   genuinely inapplicable (not merely unused)?
5. **Acceptable abstention** — for no-evidence family rows, is "cannot answer
   from the corpus" the legally correct behavior, and is the distinction
   between *no evidence exists* and *ingestion failed* the right one?
6. **Multi-contract boundaries** — are the expected contract indexes the ones a
   careful reader would draw (embedded templates, front matter, annexures)?
7. **OCR ambiguity changing meaning** — where OCR noise exists (SRC-07 prefix,
   SRC-05/06 scans), could a plausible misreading flip the expected label?

## How to record the decision

On each adjudicated row's `provenance.adjudication` (schema: `AdjudicationRecord`):

```json
{
  "status": "adjudicated",
  "adjudicator": "<real name or stable SME identifier>",
  "method": "independent_sme",
  "decided_at": "<ISO-8601 UTC instant>",
  "rationale": "<why this label wins, citing the checklist items that decided it>",
  "checklist": {"1": "pass + note", "2": "...", "3": "...", "4": "...",
                 "5": "...", "6": "...", "7": "..."}
}
```

Then set `provenance.human_review_status` to `"adjudicated"` (escalation from
`engineer_reviewed`). The validator enforces: `status=adjudicated` requires a
named adjudicator; any adjudicator string matching a model name
(`gpt|claude|gemini|llm|model`) is rejected.

## Promotion rules

- **Release requires** `adjudication.status` in `{adjudicated, not_required,
  waived_with_reason}` and completed human review (`validate_golden_set`
  condition 9); `pending_sme` blocks release.
- **Deterministic rows** (machine-verified expectations, engineer agreement)
  may use `not_required` — that is how the 64 existing release rows were merged.
- **Waiver** (`waived_with_reason`) needs a concrete recorded reason (e.g. the
  dispute is purely mechanical and reproduced by a script), decided by the
  adjudicator, never assumed.

## CI behavior

`release-quality + pending_sme` fails validation. Gate command (same as
`.github/workflows/test.yml`):

```
uv run python -m app.ai.evaluation.validate evals/golden/legal_retrieval_v2.jsonl
```

## Worked example — partial_coverage_answer-lr2-001

Query: *"Did the packet's Corrigendum I dated 25.01.2024 amend the draft
concession agreement?"* Annotator A: `relevant`; annotator B:
`partially_relevant` (the corrigendum body, SRC-04 pp333–363, is image-only).
The SME must decide which retrieval_relevance label is correct — e.g. whether
the packet index recording the fact of the amendment is enough for "relevant"
or whether the unread body limits it to "partially_relevant" — then record:
adjudicator name, `method=independent_sme`, `decided_at`, a rationale citing
checklist items 1, 3 and 7, the full checklist, `status=adjudicated`, and
`human_review_status=adjudicated`. The losing label stays in the annotations
as provenance; it is not deleted.

## Hard rules

- **Majority vote is not adjudication** for legal-interpretation disputes:
  2 engineers + 1 tie-breaker vote is not independent SME review.
- **An LLM/agent/model name is never a valid adjudicator** (validator-enforced
  via `_MODEL_ADJUDICATOR_PATTERN`); "model-assisted review" does not satisfy
  `adjudicator_must_be_human`.
- Never edit a label to match the answer key you wish were true; the expected
  value moves only with a recorded rationale.

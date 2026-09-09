# Review — rag-eval-harness

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Accepted cost — the seed golden set is a stub

`evals/golden/legal_retrieval_v1.jsonl` ships deliberately small and is not authored by a
subject-matter expert. The requirement it satisfies is **coverage of all four legal document families**
(contracts, statutes, judgments, filings), not statistical power.

This is accepted because the four changes that need a before-number are ready now, and a four-family
stub that exists is worth more to them than a comprehensive set still being negotiated. Every row is
flagged as awaiting expansion.

**What this means for anyone reading a report from it:** aggregate numbers over a handful of queries
are directional, not conclusive. A change should cite a movement, not a percentage. When the set is
expanded, its version bumps and prior reports stop being comparable — which is the whole reason the
version is carried in the report.

**Named follow-up:** expand the set with subject-matter review, bump the version, and re-record the
baseline.

---

## Accepted cost — tier-2 metrics are unavailable in this cluster

No Proof in `ingestion-chunking`, `retrieval-sql`, or `agentic-retrieval` may cite faithfulness or
answer relevancy, because the judged layer does not exist yet.

The practical consequence is sharpest for `ingestion-chunking`'s OCR engine swap: dropping EasyOCR for
RapidOCR is a change to extraction quality on the hardest document family, and tier-1 retrieval
metrics measure it only indirectly. That risk is recorded in that change rather than mitigated here.

---

## Pending — what the judged layer will add (task 5.2)

Two constraints the later tier-2 change must satisfy, recorded now so it is written as an addition:

1. **Retrieval metrics for an unchanged input stay identical.** Adding the judged layer must not
   alter a single tier-1 number for the same golden set and the same retriever.
2. **Judged metrics are a separate report section.** They are never merged into the aggregates block,
   so a reader can always tell which numbers are deterministic and which came from a judge.

_To be confirmed against the implemented report schema during 5.2._

---

## Pending — the first baseline (task 4.1)

- Golden-set version scored: _pending_
- Commit identifier: _pending_
- Aggregates, or the verbatim failure of the command that could not run: _pending_
- Retrieval path measured: the pre-`retrieval-sql` implementation. This is the number
  `retrieval-sql` and `agentic-retrieval` compare against, so it must be captured **before** either
  begins.

---

## Note — why the offline proofs cannot be trusted alone

Recorded here because it is counterintuitive and will be re-encountered by anyone extending this
harness: **tasks 1.1 through 3.3 all pass against a retriever that does not exist.** That is the
injected-protocol design working as intended, and it means the offline suite carries no information
about whether the harness is connected.

Task 3.4 is the only Proof in this change that can detect a disconnected harness. If it is skipped,
weakened, or allowed to pass on an empty result, every number this harness later produces is
unfalsifiable.

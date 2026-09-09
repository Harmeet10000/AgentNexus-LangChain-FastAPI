# Review — retrieval-sql

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured before planning — the same query, two answers

`retrieval_kb/nodes.py:237` calls `legal_rrf_search`. `DocumentQueryService.search` calls the three
branch methods and fuses them in Python. The two implementations differ in at least five ways:

| | branch path | `legal_rrf_search` |
|---|---|---|
| Base relation per leg | `chunks` | a thrice-referenced CTE, hence materialised |
| Vector query-time tuning | `SET LOCAL diskann.*` | none |
| Trigram candidate selection | ordered, then limited | `LIMIT 50` with no statement `ORDER BY` |
| Fusion constants | `constants.RRF_K` | literals `60.0` and `0.15` inline |
| Filter surface | `_FILTER_SQL` | narrower — no chunk kind, no jsonb containment, no parties |

**The agent uses the worse path.** That is the finding that made this a Class L change rather than a
tuning pass.

---

## Unverified claim, deliberately carried — the materialisation

The statement that `candidate_chunks` is materialised and therefore blocks index access is a reading of
the SQL against known planner behaviour, not a measurement. It is recorded here as unverified on
purpose.

Task 1.3 is what converts it. If the `EXPLAIN` capture shows the indexes present after all, task groups
3 and 4 shrink to the determinism and tuning fixes, and the change's ordering is unaffected — the other
four defects in the table above stand on their own.

**Do not let this claim reach implementation unmeasured.** It is the most confident-sounding sentence in
this change and the only load-bearing one that was never run.

---

## Blocking risk — the keyword access method may not exist under that name

`model.py:114` and `repository.py:418` both embed a literal access-method name. If task 1.2 finds no
such access method registered, both are wrong, and **the whole change re-scopes** — there would be no
keyword index to plan a leg over, and the keyword branch would need a different implementation
entirely.

This is why 1.2 runs second, before any seeding or planning work, and why its task text says **stop**
rather than "investigate".

---

## Possible live bug found while planning — concurrent execute on one session

`service.py:490` gathers three branch coroutines against `self.repo`'s **single** `AsyncSession`.
SQLAlchemy's async session does not support concurrent operations on one connection.

If that raises against a real connection, then **the fused path has never run in production** — every
production query has gone through `legal_rrf_search`, and the branch path's test coverage is passing
against mocks that do not exercise the concurrency.

Task 1.4 is a two-minute probe that settles it, and it runs before anything moves, because a positive
result changes task 3.2 from "route the graph at the fused path" to "serialise the branches, then route
the graph at the fused path".

---

## Accepted cost — phrase matching costs an over-fetch

The keyword extension has no phrase query, so exact phrases are implemented as over-fetch plus a
literal post-filter. A query whose phrase is rare will over-fetch and discard nearly everything.

Accepted because the alternative — a full-text vector column — would put a **third** lexical signal
into a fusion that already weights two lexical branches against one semantic branch. The same textual
evidence would be counted three times, and the fusion weights would have to compensate for a
double-count. That is a worse problem, and a subtler one.

The residual risk is the escaping. Unescaped, a phrase containing a wildcard metacharacter matches far
more than the user asked for — which is the current behaviour at `repository.py:694`.

---

## Accepted cost — the isolation ladder ships with one rung climbed

Task 5.3 adds partial indexes only. Label filtering, parallel builds, and partitioning are recorded as
a ladder with a measured trigger threshold rather than implemented.

Accepted because the corpus is empty today: choosing a partitioning strategy against zero rows would be
choosing it against a guess. What this change owes the future is the **threshold**, taken from real 5.1
recall numbers, and the recorded warning that partitioning breaks the keyword leg.

---

## Pending — the extension and access-method inventory (task 1.2)

- Extensions present, with installed versions: _pending_
- Access methods registered: _pending_
- Verdict — proceed, or re-scope: _pending_

---

## Pending — the pre-change plan capture (task 1.3)

- `legal_rrf_search`: indexes named in the plan: _pending_ (expected: none)
- keyword branch method: _pending_ (expected: the keyword index)
- vector branch method: _pending_ (expected: the vector index)
- trigram branch method: _pending_
- Seeded corpus size and tenant distribution: _pending_

---

## Pending — the concurrent-session probe (task 1.4)

- Outcome — rows returned, or a concurrent-operation error: _pending_
- If an error: task 3.2 gains branch serialisation, and this section records that the fused path had
  never executed against a real connection before this change.

---

## Pending — the tenant recall measurement (task 5.1)

- Recall for a ~1% tenant before the predicate move, against exact nearest-neighbour ground truth:
  _pending_
- Recall after: _pending_
- Plan evidence that filtering precedes the approximate scan: _pending_

---

## Pending — the measured isolation threshold (task 5.2)

- Tenant count or corpus size at which partial indexes stop being sufficient: _pending_
- The measurement this threshold was derived from: _pending_

A guessed value here fails the task. The whole point of recording a ladder is that its rungs have
numbers on them.

# Agent Saul

Tenant-aware, human-in-the-loop legal contract analysis on LangGraph. Saul ingests
messy real-world contracts into canonical, versioned chunks, retrieves evidence under
server-derived authorization, and produces grounded findings — every claim carried by a
verified evidence handle, every terminal state typed, nothing answered from memory.

📚 **Documentation:** Mintlify site in [`docs-site/`](docs-site/README.md) — `cd docs-site` and run `npx mintlify dev` locally.

![CI](https://github.com/Harmeet10000/AgentNexus-LangChain-FastAPI/actions/workflows/test.yml/badge.svg)
[![Greptile: The War on Bugs](https://www.greptile.com/badge.svg)](https://www.greptile.com/?utm_source=oss_badge&utm_medium=readme&utm_campaign=greptile_for_open_source)

## What it does

- **Bulletproof ingestion** — bounded uploads (size + extension checked before
  buffering) → queued Celery pipeline → canonical chunks carrying `contract_index`,
  `page_no`, `document_version`, `pipeline_version`. Terminal states are truthful:
  `completed`, `completed_with_warnings`, or `failed` — never a silent partial.
- **Scoped hybrid retrieval** — BM25/pg_trgm + pgvector fusion under
  `allowed_doc_ids` and `contract_index` scoping, with Graphiti as an *optional*
  relationship backend and coverage signal.
- **Evidence-handle security** — search returns server-issued, TTL-scoped handles;
  `read_evidence_by_id` re-authorizes on every call (run, thread, checkpoint,
  document version). Identity and scope never come from the model.
- **Grounded specialist analysis** — risk, compliance and deep-research sub-agents
  behind a ≤7-tool projection, mandatory grounding gate (handle membership →
  quoted-text substring check → entailment), typed findings, then human review.
- **Human-in-the-loop by construction** — plan approval and report review are
  LangGraph interrupts; resumes re-resolve authorization because a checkpoint can
  outlive a role change.
- **An evaluated system** — a provenance-gated evaluation corpus (105 cases, 23
  categories, dual-annotated, Cohen's κ 0.815) runs in CI with checksum, licensing
  and adjudication gates.

## Architecture

| | |
|---|---|
| ![Agent Saul full architecture](docs/diagrams/agent_saul_full_architecture.svg) | ![Deployment topology](docs/diagrams/agent_saul_deployment_topology.svg) |
| **Orchestration** — LangGraph with HITL interrupts, server-owned authorization, TTL evidence handles, mandatory grounding. | **Topology** — Caddy → FastAPI → role-separated Celery pools → Timescale/Postgres, RabbitMQ, Redis, Neo4j, MongoDB, R2. |
| ![Hybrid query pipeline](docs/diagrams/hybrid_query_pipeline.svg) | ![Memory stack](docs/diagrams/memory_stack_architecture.svg) |
| **Query path** — deterministic scoping → hybrid fusion → optional Graphiti enrichment → answers that disclose their evidence policy. | **Memory** — durable workflow state (checkpointer, handle store) vs advisory memory (Graphiti, cognee), surfaced through a coverage signal. |
| ![Search system overview](docs/diagrams/search_system_architecture_overview.svg) | |
| **Documents system** — bounded upload → canonical ingestion → versioned persistence → evaluation as a first-class subsystem. | |

## Honest status

What is real today: the ingestion pipeline with truthful terminal states and bounded
validation; the evidence-handle service; the Saul orchestration graph with HITL and a
grounding gate; the evaluation corpus and its CI gates (64 release-tier cases; 41
await human SME adjudication — the corpus is designated *candidate*, not gold, until
adjudication completes). What is pending: SME adjudication of judgment-dependent
labels, typed relationship edges in Graphiti (see
[PLAN-bulletproof-ingestion.md](PLAN-bulletproof-ingestion.md) Phase D), retrieval
unification and the portfolio tool layer (see [PLAN-agent-tools.md](PLAN-agent-tools.md)).

## Security model

- **Server-derived authorization.** `ToolRuntimeContext`/`AuthorizationContext`
  (user, tenant, role, `allowed_doc_ids`) is constructed from authenticated claims on
  every request and WebSocket resume — client-supplied identity is never trusted.
- **Capability-shaped evidence.** Handles are short-lived, run- and version-bound,
  and mechanically invalid outside their provenance context.
- **Untrusted documents.** Document text is data, never instructions; injection
  fixtures are part of the evaluation corpus and CI.
- **Bounded inputs.** Upload size and extension are enforced before buffering;
  per-document cost/latency budgets are part of the hardening track.

## Evaluation

The [`evals/`](evals/) subsystem is CI-gated, offline and provenance-first:

```bash
# validate a gold set (provenance, checksums, adjudication state)
uv run python -m app.ai.evaluation.validate evals/golden/legal_retrieval_v2.jsonl

# run the deterministic experiment harness (console script: legal-eval)
uv run python -m app.ai.evaluation.experiment evals/experiments/legal_retrieval_smoke.json
```

The v2 corpus (`evals/golden/legal_retrieval_v2.jsonl`, 105 cases) ships with a
per-artifact provenance manifest (source URLs, licences, SHA-256, permission and
redistribution status), dual blind annotations, and an
[adjudication runbook](evals/ADJUDICATION-RUNBOOK.md). CI fails on unknown provenance,
missing checksums, unadjudicated release-quality cases, multi-contract cases without
`contract_index`, and raw binaries whose terms disallow redistribution.

## Quickstart

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (for local
infrastructure).

```bash
git clone https://github.com/Harmeet10000/AgentNexus-LangChain-FastAPI.git
cd AgentNexus-LangChain-FastAPI
cp .env.example .env.development        # fill in providers + infra credentials

docker compose up -d rabbitmq timescale # local RabbitMQ + PostgreSQL/Timescale
uv sync --extra dev
uv run alembic upgrade head             # see caveat below

uv run uvicorn app.main:app --host 0.0.0.0 --port 5000
make celery-ingestion                   # separate terminal — ingest your uploads
```

**Migration caveat:** the Alembic chain assumes the provisioned lineage. A brand-new
database stops at revisions whose tables predate the chain (documented in
`src/alembic/versions/0012_*` and `0014_*`). If `upgrade head` fails past the
extensions, you are past what a fresh database can reach — use a provisioned snapshot.

**Tenancy note:** new users need a personal tenant before tenant-scoped routes work
(`scripts/migrate_personal_tenants.py` backfills; registration-based provisioning is
tracked in [docs/server-smoke-2026-10-28.md](docs/server-smoke-2026-10-28.md) F4).

The app also runs against fully managed infrastructure (Timescale Cloud, Redis Cloud,
MongoDB Atlas, Cloudflare R2, Railway RabbitMQ) — set the `POSTGRES_*`, `REDIS_*`,
`MONGODB_*`, `RABBITMQ_*` and `S3_*` variables in `.env.development` accordingly.

### Common commands

```bash
uv run ruff check --fix && uv run ruff format   # lint + format
uv run ty check src/                            # types
uv run pytest                                   # unit suite (integration excluded by default)
uv run pytest -m integration                    # integration (needs live services)
ast-grep scan src/                              # structural rules
make celery              # default worker pool
make celery-ingestion    # ingestion worker pool
make celery-crawler      # crawler worker pool
make celery-beat         # periodic scheduler
```

## Project structure

```
src/app/
├── api/                 # v1/v2 routers, strict response envelopes
├── ai/                  # LLM clients, embeddings, tools registry, evaluation
│   └── evaluation/      # gold-set schema, provenance validator, experiment harness
├── config/              # pydantic settings (.env.development)
├── connections/         # Postgres/Timescale, Redis, Mongo, RabbitMQ, Graphiti
├── extraction/          # Docling parsing, LangExtract (incl. two-pass relationships)
├── features/
│   ├── agent_saul/      # orchestration graph, specialists, HITL, WebSocket
│   ├── documents/       # upload, ingestion graph, retrieval, evidence service, tools
│   ├── auth/            # accounts, tenancy, legal roles, tenant migration
│   └── …                # billing, crawler, health, profile, audit
├── retrieval/           # Graphiti integration + idempotent graph writer
├── lifecycle/           # lifespan startup policies with typed degradation
└── worker/              # Celery role profiles, task catalog, beat schedule
evals/                   # golden corpus, provenance manifest, annotations, reports
docs/                    # diagrams, runbooks, smoke-test findings
docs-site/               # Mintlify documentation site
```

## Documentation

- [`docs-site/`](docs-site/README.md) — Mintlify documentation site
- [`docs/diagrams/`](docs/diagrams/) — architecture SVGs (this README embeds them)
- [`docs/server-smoke-2026-10-28.md`](docs/server-smoke-2026-10-28.md) — latest
  end-to-end server verification, with ranked open defects
- [`PLAN-bulletproof-ingestion.md`](PLAN-bulletproof-ingestion.md) — ingestion
  hardening plan (north star: the pipeline never lies)
- [`PLAN-agent-tools.md`](PLAN-agent-tools.md) — agent tool-set plan (tiny
  projection, big catalog, authority at the executor)
- [`evals/ADJUDICATION-RUNBOOK.md`](evals/ADJUDICATION-RUNBOOK.md) — how the legal
  SME adjudicates disputed labels
- [`docs/runbooks/`](docs/runbooks/) — tenant migration, upload budgets, live testing

## Acknowledgments

Built on [LangGraph](https://github.com/langchain-ai/langgraph),
[Graphiti](https://github.com/getzep/graphiti),
[LangExtract](https://github.com/google/langextract),
[cognee](https://github.com/topoteretes/cognee), [Celery](https://docs.celeryq.dev/),
[Timescale](https://www.timescale.com/) and
[Mintlify](https://mintlify.com). Public corpus documents reproduced under the
NHAI / Indian Railways / Supreme Court of India copyright policies and UK
[Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
— attribution recorded per artifact in
[`evals/corpus/`](evals/corpus/README.md).

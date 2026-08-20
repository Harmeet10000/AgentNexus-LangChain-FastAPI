I dug in properly rather than skim. Here's my honest read.
 
The headline
 
This is an unusually well-engineered skeleton of a product that doesn't exist yet. The architecture is better than most funded startups' code. The product is 0% shipped. Those two facts sit uncomfortably together, and the gap between them is the whole story.
 
What's genuinely impressive
 
Not faint praise — this is real craft:
 
- Config discipline. settings.py uses SecretStr throughout, and the PRODUCTION_SECRET_FIELDS validator refuses to boot production with default secrets. Most teams learn this after a breach.
- 178 lint/type rules in pyproject.toml, plus a migration to ty. Your git history (5c67c7d, 2beddca) shows you fixed 147 type errors and eliminated blind excepts across the codebase rather than suppressing them. That's rare discipline.
- The HITL WebSocket protocol is correct. service.py:140-186 does the real LangGraph pattern — aget_state → scan tasks[*].interrupts → emit → await → Command(resume=...), with thread-id mismatch rejection and ping/pong during human waits. People get this wrong constantly.
- The state schema. Requiring a Citation on everys the correct architectural instinct for legal AI
 
What's genuinely impressive
 
Not faint praise — this is real craft:
 
- Config discipline. settings.py uses SecretStr throughout, and the PRODUCTION_SECRET_FIELDS validator refuses to boot production with default secrets. Most teams learn this after a breach.
- 178 lint/type rules in pyproject.toml, plus a migration to ty. Your git history (5c67c7d, 2beddca) shows you fixed 147 type errors and eliminated blind excepts across the cog them. That's rare discipline.
 
The core problem
 
The Agent Saul graph is never constructed. build_saul_graph() has zero real callers — the only assignment to app.state.saul_graph is inside a module docstring at registry.py:25, and the checkpointer block in lifespan.py:294-305 is commented out. So dependencies.py:41 raises AttributeError on the first request. The router is mounted, so the endpoint exists and 500s.
 
And if it were wired, it would produce confidentl
 
# nodes.py:596-601
def _extract_risk_output(_messages: list[Any]) -> RiskAnalysisOutput:                                                            return RiskAnalysisOutput(findings=[], overal
 
The node awaits the LLM, then throws the response away. Every contract scores LOW risk, zero findings, fully compliant. For a legal risk product, a stub that returns "no risk found" is worse than a stub that crashes.
 
Also: ingestion never ingests (nodes.py:347 hardcodes text = ""), which makes normalization fail, which — because graph.py:73-74 wires unconditional edges with no failure check — silently halts the graph and emits a "done" frame with an empty summary. Every run "succeeds" with nothing in it.                                                                                     
The thing that bothers me most
 
The README says "Human gate before persistence: memory becomes trusted memory." But make_persist_memory_node(_cognee_client) takes the client with a leading underscore and nef-strings to a list. The real writers,write_clause_episodes_to_graphiti and write_final_report_to_memory, are dead code with zero callers.              
Nothing is persisted. So the gate guards nothing.
 
I don't think that's dishonesty — I think it's what happens when the README is written as a design document before the code, and then never reconciled. But right now the README describes a system and the repo contains its interface, and only one of those two is discoverable by a reader.                                                                                  name-drops "IT Act, Contract Act, SEBI".
 
You built the valuable part and left it disconnected in a sibling directory. A deterministic linter that catches a missing severability clause is more defensible in legal work than an LLM that might — it's explainable, testable, and can't hallucinate. That's your product, and it's the one thing not wired in.
 
Concrete blockers                                                                                                 
┌─────┬────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────┐
│  #  │       Issue        │                                            Evidence                                  ├─────┼────────────────────┼───────────────────────────────────────────────────────────────────┤
│ 1   │ Documents API has  │ documents/dependencies.py:62 reads request.state.user_id; nothing in src/ ever sets it. Six    │
│     │ no auth at all     │ mounted routes → 500. Fails closed, so not a data leak — but non-functional.                   │
├─────┼────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2   │ Migrations forked  │ 8a7d9b1c2e3f and a71f0d7d9c12 both have down_revision = "2bc7726317f6". alembic upgrade head   │
│     │                    │ errors. Needs alembic merge.                                                         ├─────┼────────────────────┼───────────────────────────────────────────────────────────────────┤ic that doesn't exist, on a product that has never processed one document end-to-end.
 
The tell is in logs/: one line, from nine months before your latest commit, a CORS error on first boot. This has been started once.
 
If it were mine, in this order:
                                                                                                                  1. Wire lynk in and ship it standalone. Upload a istic, cited findings. No LangGraph, no Neo4j, noially real. The engineering instincts are correct too. What's missing is one working vertical slice, and you already wrote most of it in Go.
 
---
For the chosen ones
 
Four things that put you ahead of the pack here:                                                                  
Stubs that return success are a class of bug, not an instance. _extract_risk_output returning LOW is the same failure mode as a catch {} that swallows an exception: it converts "not implemented" into "verified safe." The discipline is that every unimplemented path must fail louder than the implemented one — NotImplementedError, never a plausible default. In safety-adjacent domains this is the difference beYour _CLARIFICATION_THRESHOLD = 0.72 has the same
LangGraph users ship to production. Nodes return status=FAILED, but no edge reads status — failure is recorded in state and ignored by control flow. This is the single most common LangGraph production bug: state and control flow drift apart because add_edge is unconditional. The fix is a discipline, not a patch — any node that can set a terminal status must be followed by add_conditional_edges, no exceptions, enforced by a test that walks the compiled graph and asserts it. Your route_from_orchestrator has the mirror-image bug: it can return 10 node names but the path map lists 5, so a valid routing decision raises at runtime. Both are catchable by one graph-invariant test that needs no LLM, no DB, and about 30 lines.
t keeps you from hearing it. Delete those 20 lines, watch what explodes, and the explosion is a map of exactly where your layering broke.
 
✻ Cooked for 1m 49s
 
※ recap: You asked what I think of the scheduler/Agent Saul project, and I finished reviewing it. Verdict: strong infrastructure, but the core agent graph is never wired up and returns stubbed "no risk found" results. Next step is yours: decide whether to ship the Go lynk linter first. (disable recaps in /config)

 All nine files are in place. The repo has a large pre-existing dirty tree (those openspec syncs), so I'm
  leaving it untouched — you'll commit the new .claude/ files however you like.

  Before you test: restart Claude Code so the new agents, skill, and command load.

  What was built

  .claude/
  ├── CLAUDE.md                     ← the AGENTS.md port — now actually loads
  ├── agents/
  │   ├── scout.md                  ← Leg 1 · reads terrain, reports fog
  │   ├── planner.md                ← Leg 2 · plans at seams, read-only
  │   ├── verifier.md               ← Leg 4 · red/green verdict, raw evidence
  │   └── anchor.md                 ← Leg 5 · debt review + autonomous commit
  ├── skills/relay/SKILL.md         ← the orchestration loop you run
  └── commands/relay.md             ← /relay <task> entry point

  AGENTS.md is untouched — it stays as the OpenCode surface; CLAUDE.md is the Claude Code port.

  How it runs

          scout ──▶ planner ──▶ ┌ YOU implement ┐ ──▶ verifier ──▶ anchor
            │          │        └───────────────┘         │           │
            └──────────┴──────────── you ─────────────────┴───────────┘

  You dispatch each leg, hold every report, implement the plan yourself, and route the results. The four
  specialists are scoped lean (scout gets only read + codegraph tools), pass reports verbatim forward, and run
  once per task with a hard stop at two red laps. The anchor commits autonomously on a GREEN verdict — matching
  your choice — with blocking findings (correctness, layering breach, # noqa suppression) withholding the
  commit, and debt findings landing with a Carried forward ledger so un-fixed debt is written down. It runs git
  status + git branch --show-current first, stages named paths only (never -A), commits in Conventional
  Commits with the Co-Authored-By trailer, and never pushes.

  Try it

  /relay implement rate limiting on the auth endpoints

  You'll see each handoff announced one line before it dispatches, and the plan's Shape + Steps presented for
  your sign-off before any code is written.

  Deep Internals

  1. The four tools lists are working constraints, not decoration. The harness enforces tools: per agent —
  planner genuinely has no Write/Edit/Bash, so "a planner that could implement would stop planning and start
  typing" is enforced by the harness, not by prose. scout has Bash only for running the graphify/ast-grep CLIs;
  the moment it tried git or uv, the harness would block.
  2. The "none of them write code" property came from your own request. Your four roles are scout / planner /
  verifier / reviewer-committer — no implementer. That gave you a single-writer system, which is the strongest
  corruption-resistance available: three independent readers, one write path. If you later decide you want an
  implementing agent too, that's a real architectural change to the relay (two writers, a dispatch point,
  merge-conflict handling) — the current design would have to be extended, not tweaked.
  3. The autonomous commit interacts with your ANTHROPIC_BASE_URL proxy. Your ~/.claude/settings.json routes
  through https://agentrouter.org/ — that's what killed the two subagents I spawned earlier (405 + malformed
  response). Since your relay is subagents, that's a live risk for the whole workflow, and worth a session with
  the relay before you rely on it.

  Open items — you decide

  - Proxy health — if a relay run shows the same 405, the fix is a health-check on the router, not a config
  change on the relay.
  - graphify is dead weight in these agents — the CLI isn't installed, so the ladder falls to codegraph →
  ast-grep → rg. Fine to keep the rung; it costs nothing until the day it fires.
  - Your ANTHROPIC_API_KEY is in plaintext in ~/.claude/settings.json and transits the relay — consider
  rotating it and moving it to an env var.
  - These files are uncommitted — say the word and I'll commit just the .claude/ additions, nothing else.

✻ Churned for 6m 2s

※ recap: Your goal was analysing AGENTS.md and building a four-agent workflow; both are done, with nine files 
  created under .claude/ plus a CLAUDE.md port. Next: restart Claude Code so they load, then try /relay on a 
  task. (disable recaps in /config)
  
# Lynk V1 and V2 Split

## Purpose

This document reconciles four things into one exhaustive planning artifact:

1. the original broad "legal compiler / legal operating system" blueprint
2. the later platform and tooling expansion ideas
3. the current Lynk V1 governing plan
4. the Lynk implementation already shipped under `lynk/`

It exists to prevent loss of ideas while still keeping V1 shippable.

This is not a replacement for the governing Lynk V1 plan or the Lynk V1 LLD.
It is the bridge document that says:

- what is already done
- what still belongs in V1
- what must move to V2
- what remains in conflict and needs an explicit product decision

## Sources Reconciled

This split is derived from:

- `docs/Lynk_Plan/lynk_linter_plan.md`
- `docs/Lynk_Plan/lynk_linter_lld.md`
- the current Go implementation under `lynk/`
- the original broader blueprint for a compiler-grade legal linter, verifier, retrieval system, editor tooling, and AI-assisted legal platform

## Binding Facts Today

Unless explicitly changed by a future decision, the following are treated as binding:

1. Lynk V1 is implemented in Go.
2. Lynk V1 ships as a standalone local binary.
3. Lynk V1 is currently `.docx`-first.
4. Lynk V1 targets Indian commercial B2B contracts.
5. Lynk V1 is a deterministic linter and suggestion engine first, not a universal legal compiler.
6. Lynk V1 is CLI-first, with a reusable core for later API and editor integrations.
7. Heavy compiler infrastructure is not required for V1 unless later profiling proves it necessary.

## Current Status Summary

### Already implemented in code

- Go module under `lynk/`
- hexagonal architecture with manual constructor injection
- streaming `.docx` OOXML parser
- normalized document model
- section extraction
- definition extraction
- reference extraction
- dispute-resolution fact extraction
- deterministic rule registry
- text output renderer
- JSON output renderer
- CLI execution path
- exact rule include and exclude filtering
- severity-based non-zero exit control with `--fail-on`
- fixture-backed `.docx` parsing tests
- fixture-backed rule tests
- golden JSON CLI test
- zip bomb and archive hardening guards

### Implemented rule set

- `STR001` UnresolvedCrossReference
- `LEX005` InconsistentDefinitionCase
- `LEX012` AmbiguousTimeComputation
- `B2B019` MissingSeverability
- `B2B041` MissingGoverningLawOrJurisdiction
- `B2B052` ConflictingDisputeResolution

### Implemented test fixtures

- minimal valid contract
- conflicting dispute language
- broken cross-reference
- missing governing law
- ambiguous time period

## Decision Framework

This document classifies each capability into one of four buckets:

- `Done`: already implemented or decisively locked for current Lynk V1
- `V1`: still belongs in Lynk V1 and should be delivered before V1 is considered complete
- `V2`: intentionally deferred beyond the current Lynk V1 wedge
- `Needs Decision`: the original blueprint and current Lynk scope materially conflict, so the direction should be chosen explicitly

## Done

### Product framing and constraints

- Go selected instead of Rust or Mojo for Lynk V1
- local binary distribution selected
- `.docx` selected as the primary input format for the shipped slice
- Indian commercial B2B selected as the initial legal wedge
- deterministic core selected over LLM-first detection
- CLI-first selected over editor-first delivery
- suggestions selected as non-destructive review payloads

### Parsing and normalized model

- unzip OOXML package and stream `word/document.xml`
- use streaming XML parsing instead of building a full DOM
- merge fragmented Word runs where formatting differences are not semantically meaningful for V1
- build a normalized internal `Document`
- preserve paragraphs, sections, styles, heading level, and spans where relevant to the current model

### Current structural and semantic facts

- section index
- definition index
- reference edge list
- dispute-resolution facts

### Current linter surface

- stable diagnostic codes
- deterministic rule execution
- JSON output
- text output
- exact code selection with `--include` and `--exclude`
- severity threshold with `--fail-on`

### Current implementation architecture

- `cmd/lynk/` entrypoint
- `internal/core/model/`
- `internal/core/ports/`
- `internal/core/service/`
- `internal/adapters/docx/`
- `internal/adapters/rules/`
- `internal/adapters/output/`
- `internal/adapters/cli/`
- `internal/support/analyze/`

## V1

This section captures what still belongs in Lynk V1 without reopening the entire original compiler/platform scope.

### V1-A: Finish the current CLI and deterministic core

- deterministic diagnostic ordering
- severity-aware diagnostic sorting
- `--list-rules`
- prefix-based rule selection by code family such as `STR`, `LEX`, and `B2B`
- optionally `--quiet` if CLI UX needs it
- decide whether diagnostics should carry direct spans in V1 or remain paragraph and section based

### V1-B: Next deterministic rules already identified by the current plan

- confidentiality survival omission
- broken heading or section hierarchy
- orphaned signature structure if extractable from `.docx` layout facts

### V1-C: Minimal index growth only when justified by a concrete next rule

- heading hierarchy facts for structural validation
- signature block extraction facts
- limited section classification helpers where a rule requires them
- limited reverse reference helpers if needed for richer section-reference diagnostics

### V1-D: Narrow commercial B2B rule growth

The original broader blueprint contains many legal domains. The following still fit the current commercial B2B wedge if kept deterministic and narrow:

- `B2B089` SurvivalClauseOmission
- `B2B110` SeverabilityWithoutIntent
- `EVID08` OralModificationAllowed
- `EVID12` ParolEvidenceContradiction / EntireAgreement missing
- `TAX194` MissingTDSClause
- `TAX050` GSTIndemnification
- `B2B055` ConsequentialDamagesWaiver missing IP carve-out
- `B2B094` ChangeOfControlSilence
- `B2B102` FurtherAssurancesMissing

These should only be added if the goal is still to deepen the commercial B2B wedge rather than to start multiple domain packs at once.

### V1-E: Optional but still compatible with V1

- local whole-document content-hash cache for repeated batch analysis
- richer section-reference diagnostics such as counting downstream broken references
- limited cascading reference reporting for deleted or missing sections

## V2

This section preserves the original broader vision without forcing it into the current Lynk V1 wedge.

### V2-A: Heavy compiler infrastructure

- universal legal compiler framing across all domains
- formal legal grammar in the broad compiler sense
- CST to AST compiler pipeline as a product requirement
- red-green trees
- piece trees or other editor-grade lossless mutable buffers
- arena allocators or bump allocators as required infra
- manual memory pools
- mandatory string interning subsystem
- SIMD XML scanner or SIMD lexer as required infra
- zero-copy coordinate-pointer-only AST as the default representation
- compile-time rule generation as a required architecture choice
- bitmask-based rule configuration as a required architecture choice

### V2-B: Multi-format ingestion

- PDF
- OCR for scanned documents
- plain text and plaintext variants beyond `.docx`
- email ingestion
- WhatsApp screenshot ingestion
- generalized ingestion pipeline for mixed legal evidence and drafting inputs

### V2-C: Editor and real-time tooling

- LSP server
- VS Code extension or editor integration
- real-time incremental diagnostics
- go-to-definition for legal terms
- find references for legal terms and clauses
- hover documentation
- live code actions
- editor-first workflow
- incremental reparsing and partial invalidation

### V2-D: Verification platform and legal developer tooling

- verification engine
- test DSL
- contract test runner
- legal debugger or symbolic trace engine
- hybrid retriever combining exact, structural, and semantic search
- gRPC and HTTP APIs
- local-first bundle around the deterministic engine
- AI self-correction loop on top of the verifier
- AST-to-LLM or TOON-style compressed reasoning inputs

### V2-E: Broad legal domain expansion

- employment rule packs
- labor law packs
- RERA and real-estate-heavy registration packs
- consumer-law packs
- IP packs
- FEMA packs
- company law packs
- IT and privacy packs
- evidence law packs in full breadth
- sale-of-goods domain packs beyond the current B2B wedge
- constitutional-waiver checks
- broad Indian Contract Act legality and voidness packs

### V2-F: State-aware legal context engine

- execution state vs governing state model
- state-level inheritance and override engine
- two-phase context prefetch for late-bound governing law and jurisdiction facts
- Maharashtra override logic
- differential stamp-duty logic across states
- tri-state property, governing, and execution conflict handling
- seat-of-arbitration vs governing-state schism handling
- employment statutory-location override logic

### V2-G: Auto-refactoring and formatting

- formatter
- refactoring-safe section renumbering
- pointer rewrite on section deletion or movement
- broader auto-fix engine
- lossless formatting preservation if later justified

### V2-H: Collaboration and history platform

- immutable event sourcing for contract changes
- Git-for-law style project history
- audit-grade clause lineage
- negotiation trace storage
- retrieval across project history and prior versions

## Needs Decision

These conflicts should not be silently resolved inside the codebase. They need explicit product decisions.

### Decision 1: Is Lynk V1 still `.docx` only?

Current V1 plan says yes.
The broader platform blueprint later expands to PDF, OCR, emails, and screenshots.

Open options:

1. keep Lynk V1 `.docx` only
2. expand Lynk V1 to `.docx` and PDF
3. expand Lynk V1 to the full multi-format ingestion stack

Default until changed: option 1.

### Decision 2: Is Lynk V1 only the linter core, or also the verification platform?

Current V1 plan says linter core plus CLI first.
The broader blueprint later includes verifier APIs, retrieval, tests, editor UX, and AI orchestration.

Open options:

1. keep V1 as linter core only
2. make V1 include the verification engine API
3. make V1 include the full editor and AI loop platform

Default until changed: option 1.

### Decision 3: How broad should V1 rule coverage be?

Current V1 plan keeps the wedge narrow to Indian commercial B2B essentials.
The broader blueprint includes many more Indian legal domains.

Open options:

1. narrow V1 to the commercial B2B essentials plus immediate next deterministic rules
2. medium V1 with selected `EVID`, `TAX`, and `SGA` additions
3. broad V1 with multiple domain packs imported immediately

Default until changed: option 1.

### Decision 4: Should state-level context injection stay deferred?

Current V1 plan defers deeper state-level overrides and the execution-state vs governing-state schism.
The broader blueprint argues that this is fundamental for India.

Open options:

1. keep full state-context engine in V2
2. add minimal `ExecutionState` and `GoverningState` pointers in V1
3. make state-context injection a core V1 requirement

Default until changed: option 1.

### Decision 5: Should diagnostics carry exact spans in V1?

Current shipped Lynk slice uses paragraph and section locators.
The broader blueprint argues for compiler-grade traceability.

Open options:

1. keep paragraph and section locators only in V1
2. add span-bearing diagnostics in V1

Default until changed: option 1.

## Exhaustive Feature Inventory Split

The table below preserves the original idea inventory while assigning each item a bucket.

| Theme | Feature | Bucket | Notes |
| --- | --- | --- | --- |
| Core product | deterministic legal linter | Done | Shipped direction |
| Core product | universal legal compiler | V2 | Too broad for current wedge |
| Language/runtime | Go implementation | Done | Binding V1 decision |
| Language/runtime | Rust or Zig for maximum performance | V2 | Original exploratory argument superseded |
| Language/runtime | Mojo rewrite | V2 | Exploratory only, not current product direction |
| Parsing | `.docx` OOXML source of truth | Done | Current V1 choice |
| Parsing | stream `word/document.xml` | Done | Implemented |
| Parsing | SAX or streaming parser | Done | Implemented with Go XML streaming |
| Parsing | DOM parser avoidance | Done | Implemented |
| Parsing | run de-fragmentation merge | Done | Implemented at current V1 level |
| Parsing | PDF ingestion | V2 | Multi-format platform work |
| Parsing | OCR | V2 | Multi-format platform work |
| Parsing | email ingestion | V2 | Platform work |
| Parsing | WhatsApp screenshot ingestion | V2 | Platform work |
| Internal model | normalized document model | Done | Implemented |
| Internal model | paragraphs | Done | Implemented |
| Internal model | sections | Done | Implemented |
| Internal model | spans in model | Done | Implemented on structural nodes |
| Internal model | style facts sufficient for current rules | Done | Limited V1 form |
| Internal model | full lossless syntax tree | V2 | Deferred |
| Data structures | symbol table | Done | Present via definitions map |
| Data structures | string interning | V2 | Optional later optimization |
| Data structures | arena allocation | V2 | Deferred |
| Data structures | adjacency maps for bidirectional graph | V2 | Deferred unless a concrete V1 rule requires a small subset |
| Data structures | content-addressable cache | V1 | Good optional V1 addition |
| Semantic layer | section reference edges | Done | Implemented |
| Semantic layer | bidirectional reference graph | V2 | Deferred |
| Semantic layer | definition declaration to usage graph | V2 | Deferred |
| Semantic layer | clause-to-obligation graph | V2 | Deferred |
| Rule engine | deterministic rule registry | Done | Implemented |
| Rule engine | stable diagnostic codes | Done | Implemented |
| Rule engine | compile-time rule generation | V2 | Not required for V1 |
| Rule engine | bitmask configuration | V2 | Not required for V1 |
| Output | text output | Done | Implemented |
| Output | JSON output | Done | Implemented |
| Output | suggested fixes | Done | Implemented |
| Output | exact spans in diagnostics | Needs Decision | Current V1 uses paragraph/section locators |
| Structural rules | `STR001` unresolved cross-reference | Done | Implemented |
| Structural rules | `STR002` cascading null pointer | V1 | Narrow version still fits |
| Structural rules | orphaned signature block | V1 | Already identified as next deterministic addition |
| Structural rules | broken heading hierarchy | V1 | Already identified as next deterministic addition |
| Structural rules | execution date mismatch | V2 | Outside current wedge unless later promoted |
| Structural rules | counterparts clause missing | V2 | Deferred broader pack |
| Lexical rules | `LEX005` inconsistent definition case | Done | Implemented |
| Lexical rules | `LEX012` ambiguous time computation | Done | Implemented |
| Lexical rules | fragmented definition run forensic warning | V2 | Interesting but outside current wedge |
| B2B rules | `B2B019` missing severability | Done | Implemented |
| B2B rules | `B2B041` missing governing law or jurisdiction | Done | Implemented |
| B2B rules | `B2B052` conflicting dispute resolution | Done | Implemented |
| B2B rules | `B2B055` damages waiver carve-out | V1 | Candidate if wedge deepens |
| B2B rules | `B2B089` survival omission | V1 | Strong candidate |
| B2B rules | `B2B094` change of control silence | V1 | Candidate if wedge deepens |
| B2B rules | `B2B102` further assurances missing | V1 | Candidate if wedge deepens |
| B2B rules | `B2B110` severability without original intent | V1 | Candidate if wedge deepens |
| Employment/labor | `EMP*`, `LAB*` families | V2 | Explicitly outside current wedge |
| Real estate/RERA | `REL*`, `RER*` families | V2 | Explicitly outside current wedge |
| Company/FEMA/IP/privacy | `COA*`, `FEMA*`, `IPR*`, `ITA*` families | V2 | Explicitly outside current wedge |
| Tax and evidence | `TAX*`, `EVID*` families | V1 or V2 | Narrow selected rules fit V1; broad pack belongs in V2 |
| State context | execution state pointer | Needs Decision | Currently deferred |
| State context | governing state pointer | Needs Decision | Currently deferred |
| State context | state-schism engine | Needs Decision | Currently deferred |
| State context | Maharashtra overrides | V2 | Depends on state-context engine |
| State context | differential stamp duty | V2 | Depends on state-context engine |
| Performance | single parse then many rules | Done | Current architecture direction |
| Performance | SIMD lexing | V2 | Deferred |
| Performance | manual memory optimization | V2 | Deferred |
| Platform | CLI-first experience | Done | Current surface |
| Platform | backend API | V2 | Deferred reusable surface |
| Platform | editor/LSP | V2 | Deferred reusable surface |
| Platform | VS Code-like bundled experience | V2 | Broader platform track |
| Verification | test DSL | V2 | Platform layer |
| Verification | verification engine | V2 | Platform layer |
| Verification | legal debugger | V2 | Platform layer |
| Retrieval | exact search | V2 | Platform layer |
| Retrieval | structural search | V2 | Platform layer |
| Retrieval | semantic search | V2 | Platform layer |
| AI integration | neuro-symbolic LangGraph orchestration | V2 | Platform layer |
| AI integration | AST-to-LLM injection / TOON | V2 | Platform layer |
| Refactoring | auto-update references on renumbering | V2 | Formatter/refactor engine |
| Collaboration | immutable event sourcing | V2 | Platform layer |
| Collaboration | project history retrieval | V2 | Platform layer |

## Recommended Default Split

If no additional decisions are made, the recommended working split is:

### V1 default

- Go-based deterministic linter
- `.docx` only
- streaming OOXML parsing
- normalized document model
- sections, definitions, references, dispute facts
- current six rules plus next deterministic structural and B2B additions
- text and JSON output
- exact and prefix-based rule selection
- deterministic ordering and sorting
- optional local cache

### V2 default

- multi-format ingestion
- editor and LSP surfaces
- verification platform
- retriever and test DSL
- AI self-correction loop
- state-aware legal context engine
- broad Indian legal packs
- formatter, refactoring, and history platform features

## Why This Split Exists

The original blueprint contains two different products folded into one narrative:

1. a deterministic legal linter wedge
2. a full legal operating system with developer tooling, retrieval, editor integration, AI orchestration, and legal-domain expansion

Lynk V1 should ship the first without pretending it already contains the second.
This document exists so the second product vision is preserved, not lost, while still protecting the first from scope collapse.

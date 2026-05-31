# Lynk V1 Low-Level Design

## Goal

Define the low-level design for Lynk V1: a deterministic Go-based linter for Indian commercial B2B `.docx` contracts that ships as a single static binary and serves as the shared core for later CLI, backend, and editor surfaces.

## Scope

This LLD covers only V1 implementation details for:

- `.docx` ingestion
- normalized internal document model
- structural indexing
- deterministic rule execution
- diagnostics and suggested fixes
- CLI surface

This LLD does not cover V2 live editor features, broad legal domain packs, or formatter/autofix architecture.

## Design Constraints

1. The implementation shall be in Go.
2. The binary shall run as a standalone CLI without external runtime services.
3. The parser shall stream OOXML and avoid building a full XML DOM.
4. The linter core shall be deterministic and usable without LLMs.
5. The architecture shall preserve a reusable library boundary for future API and editor adapters.

## Module Layout

The Lynk V1 Go workspace should remain self-contained under `lynk/` and use a **hexagonal architecture with manual constructor injection**.

```text
lynk/
  cmd/lynk/
    main.go
  internal/
    core/
      model/
        document.go
        diagnostics.go
        indexes.go
      ports/
        document_source.go
        result_renderer.go
        rule.go
      service/
        lint_service.go
    adapters/
      docx/
        reader.go
      rules/
        registry.go
        structural.go
        lexical.go
        b2b.go
      output/
        text.go
        json.go
      cli/
        lint_command.go
    support/
      analyze/
        sections.go
        definitions.go
        references.go
      cache/
        document_cache.go
```

### Boundary Rules

- `cmd/lynk` handles process startup only.
- `core/model` contains domain data contracts.
- `core/ports` defines interfaces the core depends on.
- `core/service` orchestrates linting through ports only.
- `adapters/docx` implements document-source ports.
- `adapters/rules` implements rule ports and rule registry.
- `adapters/output` implements render ports.
- `adapters/cli` parses CLI args and wires constructors.
- `support/analyze` holds reusable indexing helpers used by adapters/core composition.
- `support/cache` remains optional and isolated from the core rule interfaces.

### Dependency Injection

V1 shall use manual constructor injection.

That means:

- no DI framework
- explicit wiring in the CLI adapter
- services receive interfaces and collaborators via constructors
- no package-level mutable globals for parser, rule engine, or outputs

## End-to-End Data Flow

1. CLI receives a `.docx` file path.
2. `docx.Reader` opens the zip archive and streams `word/document.xml`.
3. XML events are normalized into paragraphs, text spans, styles, and heading facts.
4. core service derives or coordinates structural indexes from the normalized document.
5. rule adapters execute selected rules over the indexes.
6. output adapters render diagnostics as text or JSON.

## Core Types

## `model.Document`

Represents normalized document state after OOXML extraction.

Responsibilities:

- canonical text view
- ordered paragraphs
- title and metadata
- source hash for future caching

Must not contain:

- rule results
- parser-only transient state
- backend-specific transport concerns

## `model.Paragraph`

Represents one normalized paragraph.

Fields should include:

- paragraph index
- normalized text
- original style name
- heading level if inferable
- source span

## `model.Section`

Represents a numbered or heading-derived section.

Fields should include:

- section number
- section title
- logical level
- paragraph membership
- source span

## `model.Indexes`

Represents derived document facts needed by rules.

Planned fields:

- sections by number
- section order
- definition map
- definition usages
- reference edges
- clause tags
- document flags

This type should be the primary input to the rule engine.

## Ports

### `DocumentSource`

Responsible for loading a `model.Document` from an input path.

Suggested shape:

```go
type DocumentSource interface {
    Read(path string) (*model.Document, error)
}
```

### `Rule`

Represents one deterministic lint rule.

Suggested shape:

```go
type Rule interface {
    Code() string
    Run(doc *model.Document, idx *model.Indexes) []model.Diagnostic
}
```

### `ResultRenderer`

Responsible for writing lint results to an output stream.

Suggested shape:

```go
type ResultRenderer interface {
    Render(w io.Writer, result model.LintResult) error
}
```

## Core Service

The core application service should orchestrate linting but remain ignorant of CLI and OOXML details.

Suggested shape:

```go
type LintService struct {
    source ports.DocumentSource
    rules  []ports.Rule
}
```

Responsibilities:

- load a document through a source port
- build indexes through the internal analysis package
- run rules in stable order
- return a `LintResult`

## `model.Diagnostic`

Represents a stable machine-readable issue.

Required fields:

- code
- severity
- message
- evidence
- paragraph or section locator
- optional suggestion

## OOXML Parsing Design

### Reader Strategy

Use `archive/zip` plus `encoding/xml.Decoder`.

The parser should:

- stream only required XML parts
- read `word/document.xml` first
- ignore unsupported OOXML branches unless needed for a V1 rule
- avoid storing raw XML trees

### Run Merging

Word fragments text aggressively across runs. The normalizer shall merge adjacent text runs into a paragraph buffer when the style boundary is not semantically meaningful for V1.

The merge policy should preserve:

- paragraph boundaries
- heading style facts

Broader run-style preservation for future formatting rules is deferred.

The merge policy may discard:

- purely decorative fragmentation irrelevant to V1 rule execution

## Normalized Text Model

V1 should use a simple append-oriented text model, not a full piece tree.

Rationale:

- V1 is batch-oriented with short-wait feedback
- `.docx` is parsed fresh per run
- live incremental editing is a V2 concern

Therefore the normalized document should be built as:

- ordered paragraph list
- canonical full-text string built once
- span offsets into the canonical text

This is simpler than an editor-grade buffer and sufficient for V1.

## Structural Analysis Design

### Section Extraction

Section extraction should combine:

- heading-style signals
- numbering regexes
- paragraph ordering heuristics

Section detection must be deterministic and transparent.

### Definition Extraction

V1 definition extraction should use conservative patterns only.

Examples:

- `"Affiliate" means ...`
- `Company`, `Customer`, `Services` in common definitional formulations

If a candidate definition is ambiguous, V1 should skip it rather than invent structure.

### Reference Extraction

Reference extraction should identify references like:

- `Section 9`
- `Section 4.2`
- `Clause 7`

The output should populate reference edges for:

- source paragraph -> referenced section

Inbound adjacency can be added later if a rule needs reverse traversal.

## Rule Engine Design

### Registry Shape

Rules should be registered as static Go values in an adapter-side registry, not loaded from external rule files in V1.

The core should depend on the rule port only.

### Rule Categories

V1 categories:

- `STR`
- `LEX`
- `B2B`

In the shipped slice, these categories are represented by stable rule-code prefixes rather than a separate category metadata layer.

### Rule Selection

CLI flags in the shipped slice support:

- include by exact code
- exclude by exact code

Likely next additions:

- include by prefix
- list rules
- quiet mode

The default remains all built-in rules.

## Initial Rule Set

### `STR001` UnresolvedCrossReference

Input:

- reference edges
- known sections

Logic:

- if a reference points to a missing section number, emit an error

### `LEX012` AmbiguousTimeComputation

Input:

- paragraph text

Logic:

- flag `N days` unless clearly qualified as `Business Days` or `Calendar Days`

### `B2B019` MissingSeverability

Input:

- full document text

Logic:

- if severability language is absent, emit a warning and suggest a standard clause pattern

### `B2B041` MissingGoverningLawOrJurisdiction

Input:

- dispute-resolution facts derived from document text

Logic:

- if no governing law or jurisdiction language is detected, emit a warning and suggest a clause pattern

### `B2B052` ConflictingDisputeResolution

Input:

- dispute-resolution facts derived from document text

Logic:

- if the document contains both arbitration language and direct court-forum language, emit a warning to review dispute-resolution consistency

## Suggestions Design

Suggestions in V1 should be non-destructive.

They should support two forms:

- exact replacement text for narrow textual fixes
- recommended clause pattern for missing-clause issues

V1 shall not mutate `.docx` output automatically.

## CLI Design

The baseline CLI command is:

```text
lynk lint [--format text|json] <document.docx>
```

The current implementation also supports:

- `--include CODE1,CODE2`
- `--exclude CODE3`
- `--fail-on warning|error`

Planned later flags:

- `--quiet`
- `--list-rules`
- prefix-based include selection

The CLI adapter should:

- parse arguments
- choose a renderer
- construct the document source and rule registry manually
- delegate to the core service
- map returned errors to exit codes

## Error Handling

### Parser Errors

- invalid zip
- missing `word/document.xml`
- malformed XML

These should terminate analysis with clear user-facing errors.

### Analysis Errors

If one derivation stage fails internally, the command should fail loudly rather than emit silent partial legal advice.

### Rule Errors

V1 rules should not panic.

If a rule hits unsupported structure, it should skip or emit no diagnostic rather than fabricate one.

## Testing Strategy

### Unit Tests

- whitespace normalization
- section heading parsing
- unresolved reference detection
- ambiguous `days` detection
- missing clause detection

### Fixture Tests

Current fixtures cover:

- valid commercial agreement skeleton
- conflicting dispute-resolution language

Next useful fixtures to add:

- broken cross-references
- missing governing law
- ambiguous time period

### Golden Output Tests

CLI JSON output now has a golden test. Expand it as more fixtures and rules are added.

## Performance Strategy

V1 performance work should focus on:

- streaming OOXML parsing
- avoiding repeated text copies
- indexing once, then running rules over indexes
- document-level concurrency only when processing multiple files

V1 should not include:

- custom memory allocators
- SIMD parsing
- fine-grained incremental reparsing

## Security Notes

- reject zip bombs with bounded archive handling
- avoid following external relationships from OOXML
- treat all document contents as untrusted input
- keep suggestion strings plain data, not executable templates

## Evolution Path

This LLD intentionally leaves room for:

- backend wrapper around the core lint pipeline
- editor/LSP adapter that reuses model and rules
- later cache and incremental invalidation work
- later Indian jurisdiction packs

## Immediate Next Implementation Steps

1. add fixtures for broken references, missing governing law, and ambiguous time periods
2. add deterministic diagnostic ordering and severity-aware sorting
3. add prefix-based rule selection and `--list-rules`
4. decide whether span-bearing diagnostics are needed in V1 or should remain deferred
5. expand indexes only when a concrete next rule needs them

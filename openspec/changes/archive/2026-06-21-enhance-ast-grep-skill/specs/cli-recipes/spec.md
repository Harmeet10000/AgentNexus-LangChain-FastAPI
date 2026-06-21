## ADDED Requirements

### Requirement: JSON output format documented

The skill SHALL document all three JSON output styles: `pretty` (default), `stream`, `compact` — with the match object TypeScript interface and examples.

#### Scenario: JSON match object schema
- **WHEN** user needs to consume ast-grep output programmatically
- **THEN** the skill shows the `Match` interface with `text`, `range`, `file`, `replacement`, `metaVariables` fields, plus the extended `RuleMatch` interface for lint rules

#### Scenario: Streaming JSON for large result sets
- **WHEN** user processes thousands of matches
- **THEN** the skill shows `--json=stream` and documents the `=` sign requirement

#### Scenario: Piping to jq
- **WHEN** user wants to filter/extract from JSON output
- **THEN** the skill shows `ast-grep run -p '$A' -r '$B' --json | jq '.[].replacement'`

### Requirement: stdin mode documented

The skill SHALL document stdin mode (`--stdin`) for piping code through ast-grep, with its caveats (no interactive mode, must specify `--lang`, must specify single `--rule`).

#### Scenario: curl + ast-grep + jq pipeline
- **WHEN** user processes remote code through ast-grep
- **THEN** the skill shows the pipe example: `curl URL | ast-grep -p 'pattern' --json --stdin -l html | jq ...`

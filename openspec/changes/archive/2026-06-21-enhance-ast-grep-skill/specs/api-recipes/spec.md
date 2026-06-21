## ADDED Requirements

### Requirement: JavaScript API recipes documented

The skill SHALL document 3 common JavaScript API patterns: find, findAll, and complex rule using `NapiConfig`, with `@ast-grep/napi`.

#### Scenario: Find first match
- **WHEN** user needs to find the first AST node matching a pattern
- **THEN** skill shows `root.find('console.log($A)')`

#### Scenario: Find all matches
- **WHEN** user needs to iterate all matching nodes
- **THEN** skill shows `root.findAll('console.log($$$)').forEach(n => n.text())`

#### Scenario: Complex rule with NapiConfig
- **WHEN** user needs to pass a YAML-like rule object from JS
- **THEN** skill shows `root.find({ rule: { pattern: '...', ... } })`

### Requirement: Python API recipes documented

The skill SHALL document 3 common Python API patterns: parse, find_all, and accessing matched variables.

#### Scenario: Parse and find
- **WHEN** user needs to parse source code in Python
- **THEN** skill shows `from ast_grep_py import parse; ast = parse("typescript", source)`

#### Scenario: Access matched variables
- **WHEN** user needs to extract captured meta variables
- **THEN** skill shows `node.get_match("A").text()`

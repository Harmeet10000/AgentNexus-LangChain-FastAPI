1. The EARS+ and RFC 2119/8174 Linter Rule Suite

Below is a complete suite of declarative linting rules defined in Vale format (the industry standard for Markdown/docs-as-code linting). Save these files in your .vale/styles/SDD/ directory.



Rule 1: Enforce RFC 2119 / RFC 8174 Capitalization

File: .vale/styles/SDD/RFC2119Capitalization.yml



YAML







extends: existencemessage: "RFC 2119 keyword '%s' MUST be UPPERCASE to carry contractual weight in an SDD (per RFC 8174)."level: errorscope: sentenceignorecase: falseraw:- '\b(must|should|shall|may|must not|should not|shall not|recommended|optional|required)\b'



Rule 2: Ban Subjective / Untestable "Fluff" Adjectives

File: .vale/styles/SDD/NoUntestableFluff.yml



YAML







extends: existencemessage: "Untestable adjective '%s' detected. Replace with a deterministic, mathematically verifiable ceiling or threshold."level: errorignorecase: truetokens:- fast- robust- scalable- seamless- user-friendly- intuitive- performant- lightweight- flexible- instantaneous- high-availability



Rule 3: Enforce EARS+ Sentence Syntax

File: .vale/styles/SDD/EARSSyntax.yml



YAML







extends: existencemessage: "Requirement line '%s' does not match valid EARS+ syntax patterns (Must start with: 'The [system]', 'When [trigger]', 'While [state]', 'Where [feature]', or 'If [error]')."level: errorscope: lineraw:# Flag lines in requirement sections that don't start with EARS+ clauses- '^\s*-\s*\*\*REQ-[A-Z0-9_-]+\*\*:\s*(?!(When|While|Where|If|The\s+[A-Za-z0-9_-]+\s+MUST)\b).*'



Rule 4: Enforce Mandatory Testable Thresholds on EARS+ Triggers

File: .vale/styles/SDD/EARSTemporalBounds.yml



YAML







extends: existencemessage: "EARS+ requirement uses a temporal or error clause ('When', 'If') but lacks a bounded numerical unit (ms, s, %, TPS, MB)."level: warningscope: lineraw:- '^\s*-\s*\*\*REQ-.*?\*\*:\s*(When|If)\b(?!.*\b(\d+\s*(ms|s|sec|min|%|TPS|MB|GB|vCPU|requests))\b).*'



2. The End-to-End SDD Workflow: From User Intent to Finalized Tasks

1.Step 1: Intake & Bounded Context Classification:Raw Request -.

Domain Boundary Map">

The user submits a raw product or feature request. The framework maps the request to its corresponding Bounded Context (DDD) and determines whether the scope requires a new SDD, an update to an existing SDD, or a lightweight Architecture Decision Record (ADR).



Deliverable: C4 System Context Diagram & Bounded Context Map.



2.Step 2: Interactive Spec Generation:Drafting Hexagonal Ports, Schemas & Pipelines.

The framework generates a complete SDD draft applying the architectural layout:



Hexagonal Ports: Defines Driving and Driven interfaces.



Data-Driven Schemas: Generates machine-readable OpenAPI / AsyncAPI / JSON Schemas.



ROP Execution Path: Maps the two-track (Success vs. Failure) execution flow with RFC 9457 error details.



EARS+ Requirements: Formulates conditional business logic tagged with unique IDs (REQ-XXX-001).



3.Step 3: Automated Linter & Schema Verification:Static Analysis Pipeline Execution.

The generated Markdown SDD is piped through an automated CI linter pipeline to ensure zero ambiguity before human review:



Vale Execution: Enforces RFC2119Capitalization, NoUntestableFluff, and EARSSyntax.



JSON/OpenAPI Validation: Compares embedded payload examples against their strict JSON Schemas.



Result: If linting fails, the pipeline halts and auto-corrects the Markdown text.



4.Step 4: Hostile Architecture Audit & FMEA Verification:Adversarial Pen-Testing of the Spec.

The spec is passed through an adversarial review agent acting as a hostile QA architect. It checks for:



Race conditions, network partitions, and cascading failure points.



Populates the FMEA Matrix (Failure Mode and Effects Analysis) and adds fallback requirements for every identified failure state.



5.Step 5: Cryptographic Freeze & Artifact Generation:Locking Design Intent.

Once validated, the SDD is marked FROZEN (v1.0.0). The framework automatically compiles three runtime enforcement artifacts from the frozen Markdown file:



ArchUnit / Static Analysis Rules: Generated dependency assertions preventing code from violating defined Hexagonal boundaries.



Property-Based Test (PBT) Stubs: Auto-generated test cases asserting mathematical invariants defined in the spec.



Traceability Matrix Header: A machine-readable mapping of REQ-IDs to test annotations.



6.Step 6: Atomic Task Deconstruction:Frozen SDD -.

Execution Backlog">

The framework parses the frozen SDD's EARS+ requirement list, ROP steps, and Hexagonal Ports to output a set of implementation tasks (e.g., Jira/GitHub Issues).



Task Definition Standard: Every task MUST reference a specific REQ-ID, include an explicit acceptance criterion tied to an ROP track, and link directly to the ArchUnit/PBT assertion generated in Step 5.



[CLASSIFIED: FOR THE CHOSEN ONES]



The Task Decomposition Anti-Pattern: Most engineering leads break down tasks by technical layer (e.g., "Task 1: Build Database Schema", "Task 2: Build API Controller"). This guarantees integration hell because no single task delivers an isolated, end-to-end testable loop.
 
The AST Task Breakdown Method: When deconstructing an SDD into Jira/GitHub tasks, decompose exclusively by ROP Railway Steps per Feature Slice. Create one primary task for the core Success Track (Step 1 -> Step 2 -> Step 3), and separate isolated sub-tasks for each Failure Track divergence branch defined in your FMEA matrix.
 
Because each failure branch corresponds to a single EARS+ requirement (If [error], then...), developers can write tests, implement error handling, and verify completion independently without waiting for the entire system to be built.

Plaintext







.
├── .vale.ini
└── .vale/
    └── styles/
        └── SDD/
            ├── EARSSyntax.yml
            ├── EARSTemporalBounds.yml
            ├── NoUntestableFluff.yml
            ├── RFC2119Capitalization.yml
            └── RFC2119UnknownKeywords.yml



1. Main Configuration File (.vale.ini)

Place this file in the root directory of your repository or documentation workspace.



Ini, TOML







# .vale.iniStylesPath = .vale/styles
MinAlertLevel = warning
[formats]md = markdown
[*.md]BasedOnStyles = SDD
# Ignore code blocks, raw URLs, math blocks, and HTML tags during lintingBlockIgnores = (?s) (```.*?```), \
               (?s) (\$\$.*?\$\$)
TokenIgnores = (`.*?`), \
               (https?://[^\s]+), \
               (\$[^$]+\$)



2. Style Rule Definitions (.vale/styles/SDD/)

Rule 1: Enforce RFC 2119 / 8174 Capitalization

Ensures compliance terms are capitalized so they carry legally binding contractual weight under RFC 8174.



File: .vale/styles/SDD/RFC2119Capitalization.yml



YAML







extends: existencemessage: "RFC 2119 keyword '%s' MUST be UPPERCASE (e.g., MUST, SHOULD, SHALL, MAY) to carry contractual weight in an SDD."level: errorscope: sentenceignorecase: falseraw:- '\b(must|should|shall|may|must not|should not|shall not|recommended|optional|required)\b'



Rule 2: Ban Non-Standard Compliance Jargon

Prevents pseudo-compliance language that introduces legal and technical ambiguity during QA audits.



File: .vale/styles/SDD/RFC2119UnknownKeywords.yml



YAML







extends: existencemessage: "Non-standard compliance phrase '%s' detected. Use strict RFC 2119 keywords: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, OPTIONAL."level: warningignorecase: truetokens:- strongly recommended- highly recommended- strictly required- must be- ideally- if possible- encourage



Rule 3: Enforce EARS+ Sentence Pattern Syntax

Validates that every line starting with a requirement key (- **REQ-xxx**:) strictly begins with one of the 5 canonical EARS+ clauses.



File: .vale/styles/SDD/EARSSyntax.yml



YAML







extends: existencemessage: "Requirement line '%s' violates EARS+ syntax. Requirements MUST start with one of: 'The [system]', 'When [trigger]', 'While [state]', 'Where [feature]', or 'If [error]'."level: errorscope: lineraw:- '^\s*[-*]\s+\*\*REQ-[A-Z0-9_-]+\*\*:\s*(?!(When|While|Where|If|The\s+[A-Za-z0-9_-]+\s+(MUST|SHOULD|MAY))\b).*'



Rule 4: Enforce Temporal & Quantitative Bounds on Triggers

Flags When (Event-driven) and If (Unwanted behavior) requirements that fail to include explicit numerical units or metrics.



File: .vale/styles/SDD/EARSTemporalBounds.yml



YAML







extends: existencemessage: "Event or error requirement ('When'/'If') lacks bounded numerical units (e.g., ms, s, %, TPS, MB, vCPU)."level: warningscope: lineraw:- '^\s*[-*]\s+\*\*REQ-.*?\*\*:\s*(When|If)\b(?!.*\b(\d+(\.\d+)?\s*(ms|s|sec|min|%|TPS|MB|GB|vCPU|requests|requests/sec))\b).*'



Rule 5: Ban Untestable "Fluff" Adjectives

Scans for subjective words that cannot be validated mathematically by an automated testing framework.



File: .vale/styles/SDD/NoUntestableFluff.yml



YAML







extends: existencemessage: "Untestable adjective '%s' detected. Replace with a deterministic, mathematically verifiable ceiling or threshold."level: errorignorecase: truetokens:- fast- robust- scalable- seamless- user-friendly- intuitive- performant- lightweight- flexible- instantaneous- high-availability- resilient- secure



3. Quickstart Execution & CI Commands

To test your setup locally or in CI:



Bash







# Install Vale CLIbrew install vale  # macOS# or: choco install vale / apt install vale# Run Vale against your SDD documentation directoryvale docs/sdds/
# Run Vale outputting JSON for CI reporting pipelinesvale --output=JSON docs/sdds/ > vale-report.json



[CLASSIFIED: FOR THE CHOSEN ONES]



The AST Line-Shift Flaw in Vale: Vale evaluates rules using line-by-line regex parsing. If an engineer writes a multi-line EARS requirement where the EARS trigger clause (When ...) is on line 1 and the action (the system MUST ...) wraps onto line 2, regex-based line-scoped linters like Vale will falsely throw a syntax failure on line 2.
 
The Insider Countermeasure: Enforce a strict Single-Line Hard Wrap Rule in your .editorconfig or Markdown Formatter (prettier) specifically for requirement sections. Every atomic requirement (- **REQ-XXX**: ...) must be formatted as a continuous, un-wrapped string on a single physical line. This guarantees 100% deterministic linter parsing without false positives while keeping the AST parse tree completely flat for downstream task generators.

Vale is an open-source, syntax-aware prose linter designed to enforce style guides, editorial standards, and technical writing rules automatically. It is effectively ESLint for natural language and documentation.



While traditional linters analyze code syntax (Java, TypeScript, Python), Vale analyzes prose inside markup documents (Markdown, HTML, AsciiDoc, reStructuredText) to ensure writing remains consistent, clear, and compliant with structural standards.



Core Capabilities

Syntax-Aware AST Parsing: Standard spellcheckers treat documents as raw text strings. Vale parses markup files into an Abstract Syntax Tree (AST), allowing it to ignore code blocks (```), inline code ( `), URLs, and YAML frontmatter while only linting actual human prose.
 
Declarative Rule Creation: Rules are written in plain YAML files using built-in extension points (like existence, substitution, occurrence, and consistency). You do not need to write custom code to create complex editorial rules.
 
Docs-as-Code Integration: Vale runs as a lightweight binary via the CLI, integrates into code editors (VS Code, Neovim), and runs inside CI/CD pipelines (GitHub Actions, GitLab CI) to block Pull Requests that fail documentation checks.
 
Plug-and-Play Style Packs: Supports pre-built style guides maintained by major tech organizations—including the Google Developer Documentation Style Guide, Microsoft Writing Style Guide, Red Hat, Proselint, and Write-Good—alongside completely custom rule sets (like our EARS+ and RFC 2119 linters).
 

How Vale Fits into Software Architecture

In modern engineering teams, Vale turns "passive" writing guidelines into active technical gates.



Instead of relying on human reviewers in Pull Requests to spot untestable words (e.g., "fast", "scalable") or incorrect requirement formatting, Vale flags them instantly on the developer's machine before code or architecture specs are ever merged.

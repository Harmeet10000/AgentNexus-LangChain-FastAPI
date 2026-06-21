## ADDED Requirements

### Requirement: Comprehensive FAQ with common problems

The skill SHALL include an FAQ section with at least 15 entries covering: pattern not matching, shell escaping, language detection, meta variable capture issues, slow search, rule not found in scan, test discovery, strictness tuning, inline suppression, multi-file rules, custom languages, JSON output format, API usage, editor integration, and performance.

#### Scenario: Pattern not matching
- **WHEN** user's pattern does not find expected code
- **THEN** FAQ provides steps: check language, check strictness, try `--debug-query`, use playground

#### Scenario: Shell escaping
- **WHEN** `$VAR` is being expanded by shell
- **THEN** FAQ states: always use single quotes `'$PATTERN'`

#### Scenario: Language detection issues
- **WHEN** ast-grep doesn't detect the language correctly
- **THEN** FAQ suggests using `--lang` explicitly or configuring `languageGlobs`

#### Scenario: Rules not firing in scan
- **WHEN** `ast-grep scan` doesn't report expected issues
- **THEN** FAQ suggests checking `ruleDirs`, `sgconfig.yml`, `files`/`ignores`, and using `--inspect`

## ADDED Requirements

### Requirement: SecretStr pattern documented
`.opencode/instructions/PYTHON-TYPING-RULES.md` SHALL document that secrets in `Settings` use `pydantic.SecretStr` and consumers call `.get_secret_value()`.

#### Scenario: Documentation includes example
- **WHEN** the instruction file is read
- **THEN** it SHALL contain a concrete example showing `SecretStr` field declaration and the correct `.get_secret_value()` usage pattern

### Requirement: PrivateAttr pattern documented
`.opencode/instructions/PYTHON-TYPING-RULES.md` SHALL document when to use `PrivateAttr` — only for non-serializable runtime state derived from validated fields.

#### Scenario: Documentation includes PrivateAttr rules
- **WHEN** the instruction file is read
- **THEN** it SHALL state: use `PrivateAttr` only for runtime-only, non-serializable state; prefer `@property` or `Field(exclude=True)` for serializable values

### Requirement: Pydantic v2 Field patterns documented
`.opencode/instructions/PYTHON-TYPING-RULES.md` SHALL include guidelines for `Field(default_factory=...)` for mutable defaults, `ConfigDict(frozen=True)` for immutability, and `extra="forbid"` for request models.

#### Scenario: Documentation covers Field patterns
- **WHEN** the instruction file is read
- **THEN** it SHALL contain rules for `default_factory`, `frozen=True`, `extra="forbid"`, and `SecretStr`

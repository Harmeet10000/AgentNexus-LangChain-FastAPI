# Glossary Skill Reference

The `glossary` skill helps teams keep project terminology consistent across
proposals, requirements, specifications, design documents, ADRs, and task plans.

This page explains the files the skill uses and the output you should expect to
see when it is applied.

## Glossary Files

Glossary files live at the project root:

| File | Purpose |
| --- | --- |
| `glossary/business.md` | Domain, product, workflow, and business terms. |
| `glossary/technical.md` | Architecture, implementation, platform, and protocol terms. |

Example:

```markdown
| Term | Definition | Use When | Avoid |
| --- | --- | --- | --- |
| Account | A registered customer identity that owns profile, billing, and access settings. | Referring to the customer identity in product workflows. | Using for a bank account, login session, or internal staff user. |
```

Use glossary entries for project-specific language. Avoid generic dictionary
words and one-off implementation details.

## Companion Reference Files

When an artifact uses glossary terms, it gets a companion reference file in the
same directory. Remove `.md` from the artifact name and append
`-glossary-reference.md`.

| Artifact | Companion reference |
| --- | --- |
| `proposal.md` | `proposal-glossary-reference.md` |
| `design.md` | `design-glossary-reference.md` |

The companion file lists which glossary terms the artifact uses, where each term
came from, and how the artifact uses it. It does not copy definitions.

```markdown
# Glossary Reference

| Term | Source Glossary | Context |
| --- | --- | --- |
| Account | `glossary/business.md` | The customer identity that owns the subscription being changed. |
```

If no glossary terms are used, the companion file says so:

```markdown
# Glossary Reference

No glossary terms referenced.
```

## Marking Terms In Artifacts

Glossary terms that appear in a companion reference file are bolded in prose in
the artifact itself.

Example:

```markdown
During checkout, the **Account** owner can update the subscription plan.
```

Do not bold terms in code blocks, frontmatter, links, or places where Markdown
formatting would reduce readability.

## Framework Usage

The glossary skill is framework-agnostic. It can be used with any documentation,
specification, planning, or requirements workflow where terminology consistency
matters.

For OpenSpec projects, require it from `openspec/config.yaml`:

```yaml
rules:
  proposal:
    - Must use glossary skill
```

You can also require it for other OpenSpec stages, such as `design`, or
`spec`, when those artifacts should be checked against the glossary.

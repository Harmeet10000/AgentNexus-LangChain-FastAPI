# Evaluation and regression checks

Run from the skill directory:

```bash
uv run --no-project --with PyYAML --with markdown-it-py --with mdit-py-plugins python -m unittest discover -s scripts -p 'test_*.py'
```

The parser and preservation tests cover valid extensions, unsafe/duplicate YAML, Markdown footnotes, code fences, trust records, reserved filenames, encoded local links, Unicode/CRLF, empty inputs, source mutation, and coverage gaps. Reindex tests cover manual indexes, idempotence, and dry-run nonmutation.

`evals/evals.json` defines outcome cases and `evals/triggers.json` defines activation expectations. They are evaluation inputs, not fabricated successful runs. When a compatible agent harness is available, run matched isolated cases with and without this skill; keep outputs and traces outside the package. Review semantic quality and inspect mistaken tool use. Record version, actual run count, pass evidence, and timing when available. Do not count a missing or failed run as a negative model result.

Acceptance for a curation run: unchanged original digests; exact source reconstruction; every source mapped to a read record and new topic; no authored local links to originals in the standalone collection; current specification metadata; primary-source citations for researched factual additions; a final passing report after the final reindex.

## Executed independent workflow check

A GPT-5.6 Luna agent used the skill in an isolated fixture workspace containing Unicode CRLF text, unrelated topics, and an inherited unresolved link. It preserved the original bytes, created separate topics, ran the validator and reindexer from another working directory, and confirmed that a deliberately broken authored link caused strict validation to fail. The repaired fixture passed. See [the actual result](../evals/forward-test-result.json). This is one executed workflow check; it does not claim the proposed trigger dataset or all broader evaluation cases have been run, and it does not evaluate factual enrichment quality.

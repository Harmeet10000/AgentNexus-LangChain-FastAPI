# Applied Agent Skills guidance

Consulted 2026-09-12. The final authenticated Firecrawl CLI search and bounded crawl succeeded: eight URLs cover five distinct documentation topics, including HTML/Markdown variants. Earlier direct-HTTPS fallback evidence is retained separately. The CLI status renderer omitted page content, so the full completed job response was saved using the CLI’s bundled SDK. See [retrieval provenance](firecrawl-retrieval.json).

| Official source | Application in this package |
| --- | --- |
| [Best practices](https://agentskills.io/skill-creation/best-practices) | Keep the main procedure focused; move optional detail into named references; preserve concrete failure lessons from this corpus build. |
| [Using scripts](https://agentskills.io/skill-creation/using-scripts) | Noninteractive flags, inline dependencies, JSON output, meaningful exits, report overwrite protection, and a reindex preview. |
| [Evaluating outputs](https://agentskills.io/skill-creation/evaluating-skills) | Regression fixtures exercise preservation and parser failure cases; outcome cases remain distinct from tests of script behavior. |
| [Optimizing descriptions](https://agentskills.io/skill-creation/optimizing-descriptions) | The description names user intentions and includes mixed notes, migration, and validation; near-miss prompts test scope. |
| [Specification](https://agentskills.io/specification) | Lowercase installable `okf` directory, matching skill name, explicit compatibility, string-valued metadata, and conditionally loaded references. |

No claim of improved model trigger rate or reduced token use is made: that requires recorded independent runs. Deterministic script tests establish script behavior, not the quality of a model's future prose.

## Refinements from the completed crawl and execution

* Preserve exact bytes with deterministic tools; leave topical grouping flexible. This concentrates prescriptive steps where loss is hard to detect.
* Review execution traces as well as final reports. The isolated workflow exercised CRLF, source reconstruction, another working directory, and a deliberately rejected authored link.
* Keep claimed gains separate from functional checks. No baseline model comparison, trigger-rate measurement, or token-efficiency improvement is inferred from this run.
* Inspect saved research content, not just a successful job status. CLI 1.18.5 can report a completed crawl without including its page bodies in the formatted status output.

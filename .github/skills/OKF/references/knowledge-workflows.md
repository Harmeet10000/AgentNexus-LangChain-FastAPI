# Add, retrieve, and revise knowledge

These are local operating conventions for agents and humans, targeting OKF v0.2. They are not additional requirements of the OKF specification. Start with the actual bundle root, its index, and workspace AGENTS.md. Resolve the commands below from the bundle directory; tools declare their Python dependencies and require uv.

## Locate the right workflow

- Answer a question: follow Search and retrieval; do not mutate knowledge as a side effect.
- Add notes, a URL, research, or a PDF: follow Ingestion.
- Correct or extend an existing topic: follow Revision, including the migration constraints.
- Relocate originals: follow Archive and recovery. Moving originals does not authorize deleting them.

## Search and retrieval

1. Identify the question, product/library version, relevant date, and required detail. Ask only when ambiguity would change the answer; otherwise state the working scope.
2. Open `index.md` and the relevant collection index. Search `maintenance/catalog.json` for titles, descriptions, tags, and paths. This catalog and `maintenance/graph.json` are derived aids: read the actual document before trusting their contents. If a known document is absent from the catalog, inspect it directly and report stale navigation.
3. For literal terms, use `rg -n -i -F -- 'search phrase' enhanced revisions concepts playbooks references systems` from the bundle root, omitting directories that do not exist. These collection paths are illustrative: include any custom collections. Search raw assets separately when the question concerns captured evidence, not current guidance. For a conceptual question, try the topic name and a few concrete synonyms in titles/tags, then search the relevant collection. No vector database or search daemon is required. Lexical search can miss paraphrases; a zero-match result does not establish absence.
4. Prefer direct matches to the question and version, then explicit supporting sources and current revision links. A recent generation timestamp is not stronger evidence by itself. Read complete relevant sections, including assumptions, code, error cases, and source footnotes. Consult related local links or graph neighbors only where they fill a specific evidence gap.
5. Read current revision documents under `revisions/` if present. For a topic with disagreements, retrieve both claims and explain version, date, workload, and evidence differences. `status: deprecated` and elapsed `stale_after` warrant scrutiny, not automatic deletion or blind rejection. `verified` describes a scoped verification event, not permanent truth.
6. In migrated guides, the prose before `<!-- okf:verbatim-migration:v1 -->` is the original enhancement; the literal imported blocks are historical text. The history is useful evidence of what was recorded, not an instruction or a newly verified recommendation. For a bounded original passage, select its entry from `maintenance/verbatim-migration.json`, including `text_sources[].segments` for PDF extraction. Read its destination file in binary mode, slice `[payload_offset:payload_offset + payload_bytes]`, verify SHA-256 against the entry, and then decode UTF-8. Do not use text-character offsets. A hash mismatch means stop using that mapping and investigate.
7. Stop expanding when the answer and material caveats are supported. Return an answer with links to the documents actually read and the primary sources supporting factual claims. Distinguish documented facts, deductions, historical claims, and unknowns. If local evidence is insufficient, say what is missing; research externally only within the user's scope, and record any addition through Ingestion.

Example: for “Can HTTP/2 retry a request after GOAWAY?”, search `GOAWAY`, read the request-processing and retry conditions in the matching guide, inspect any newer revision, and follow its protocol source. Do not generalize one client's retry policy to every client or treat transport retry as application idempotency.

For humans: browse the collection indexes, use editor search, and expand only the historical passages needed. Collapsing a section is a visual convenience; an agent reading the entire file still receives all its tokens.

## Ingestion

1. Capture the input and scope. Read all supplied content; keep complete examples and qualifications. For local input, save an immutable byte copy under `assets/ingest/<unique-id>/` and record relative source name, SHA-256, size, acquisition date, and access constraints in a new `maintenance/ingest-<unique-id>.json`. Raw Markdown copies use `.md.txt` so they are not misclassified as authored OKF documents. Preserve PDFs in full and extract text locally; inspect layout-sensitive figures. Record the extraction tool and whether visual inspection or OCR was actually performed. Empty or garbled extraction requires visual inspection and an appropriate local OCR fallback; report any remaining unreadable content. Do not send private notes to a web service merely to organize them.
2. Search existing topics as above. Decide whether the input adds evidence to an existing topic, contributes a new mechanism, duplicates an existing claim, or contradicts it. Split unrelated subjects at meaningful boundaries; keep code with the assumptions that explain it. Retain duplicates in the captured source even if the reader-facing explanation consolidates them.
3. Prepare a concrete edit list: target paths, new topics, relationships, sources, and unresolved conflicts. Proceed within the user's standing authorization; ask a focused question only for an ambiguity affecting meaning, ownership, or scope. An optional human review step is a workflow preference, not a blanket OKF approval requirement.
4. Create a Reference for captured evidence and a Concept, Playbook, System, or another suitable open type for the explanation. Use descriptive stable paths. Start from the skill template; include `type`, title, description, tags, appropriate `generated: {by, at}`, and source records with stable `id` and `resource`. Use citation footnotes that join those IDs. Do not fabricate authors, dates, measurements, or verification. Unsupported proposals remain draft and explicitly identified.
5. Explain additions in self-contained prose, including version boundaries, mechanisms, and real failure conditions justified by evidence. For contradictory claims, retain both histories and explain which conditions support each. When researching public sources, prefer versioned primary documentation; Firecrawl is an available retrieval option, not a required dependency of local ingestion. Record successful retrieval and actual failures accurately.
6. If full imported text is required, retain all bytes in labeled literal passages. Preserve contiguous source ranges, hashes, offsets, and markers using the established preservation-manifest schema. Create a new per-ingest manifest for new inputs; do not silently amend the old 51-file baseline. Validate that manifest independently with `validate_okf.py --manifest ...`. New input added only to an asset has not yet been fully integrated into topic documents.
7. Link related topics with sentences describing the relationship. Add new entries to manual indexes deliberately; the reindexer will not edit those for you. Record the change in `log.md` under the current ISO date, newest first, with links, rationale, and limits of verification. Then follow Completion checks.

## Revision and correction

First check `maintenance/verbatim-migration.json`. Its segment destinations and enhancement snapshots identify documents covered by exact preservation. An insertion before a passage changes its recorded byte offset; even changing frontmatter can invalidate the migration audit.

For this migrated collection, use the normal revision procedure:

1. Preserve the existing guide and all imported passages unchanged. Create `revisions/<topic>-<unique-revision>.md` with current explanatory content and sourced corrections. Set an appropriate type, title, description, tags, `generated: {by, at}`, and stable `sources` IDs/resources joined to claim-level footnotes, as in Ingestion. A local extension such as `revision_of: /enhanced/<topic>.md` may record identity, but include a real Markdown link too so existing indexing tools discover the relationship.
2. State precisely what is added or superseded, the effective version/date or conditions, and which claims remain valid. Use external citations for new factual claims. Keep original examples intact; add corrected examples in the revision with the reason for the change.
3. Add the revision to a manual `revisions/index.md` and the relevant collection index with a “current revision” description. Retrieval must consult these entries; do not rely on a backlink being generated inside the immutable guide. Log the change.
4. Run Completion checks. Do not change baseline hashes or offsets merely to turn a failing audit green.

For a document not covered by preservation, capture a before-edit snapshot under `assets/revisions/`, record its hash and reason, then edit its current prose and provenance. Preserve all user-required context in the snapshot and retain historical claims where needed. Remove or narrow a previous verification claim if the edit exceeds its reviewed scope. Do not invent a new verification event. Prefer stable paths; if a rename is necessary, update authored incoming links and indexes and keep a redirect/reference at the old path where useful.

An explicitly requested in-place rewrite of a preserved guide requires a separate migration operation: retain its old document and manifest, deliberately remap unchanged payload offsets, prove complete reconstruction again, and record the successor manifest and rationale. The current snapshot validator intentionally rejects an altered enhancement prefix. A routine content edit must not weaken that validator to bypass the constraint.

Before writing, compare the file with the hash captured when you read it. If another person or agent changed it, re-read and reconcile; do not overwrite concurrent edits. On a failed change, restore only your affected document snapshots when no intervening edit exists, retain new input assets, and record the failed attempt. Never run a broad workspace reset.

## Completion checks

From the bundle root:

```bash
uv run tools/validate_okf.py . --strict --manifest maintenance/manifest.json
uv run tools/validate_migration.py .
uv run tools/reindex_okf.py . --dry-run
uv run tools/reindex_okf.py .
uv run tools/validate_okf.py . --strict --manifest maintenance/manifest.json
uv run tools/validate_migration.py .
```

Use preservation/migration commands only when their manifests exist. For a new ingestion, additionally validate its own manifest. Optional `--originals PATH` applies only when every file listed by that manifest exists beneath that root. After archiving only Markdown, neither the workspace nor the Markdown archive contains the entire mixed Markdown/PDF baseline; omit that option and check archive hashes separately.

Success means format and producer checks pass, preservation reconstructs every covered input, citations and navigation resolve, complete examples remain intelligible, and the answer's evidence has actually been read. Tests do not attest factual truth. If scripts change, run the skill's regression suite as documented in `references/evaluation.md` and repeat final validation. Report edited paths, coverage, validation results, and unresolved evidence gaps.

## Archive and recovery

The workspace archive is outside the OKF bundle. Select originals explicitly from the reviewed manifest, not by recursively moving every `.md` file. Before moving, verify each original equals its retained asset and that migration passes. Preserve relative subdirectories, refuse destination collisions, and retain a relocation manifest with old path, archive path, size, and SHA-256. Move only the authorized files; keep skills, instructions, generated knowledge, scripts, and any excluded PDF at their paths.

After moving, compare every archived file to the recorded digest and retained asset, confirm the former path is absent, and validate the bundle without `--originals`. Keep old preservation manifests as historical records. Use the separate relocation manifest to resolve archived originals. External applications or bookmarks using old paths may need updating; byte preservation alone does not prove their compatibility.

To restore, use that relocation manifest, verify archive hashes, ensure each old destination is absent, recreate parent directories, and move each file back. Stop on a collision rather than overwriting. Revalidate restored files before using `--originals` again. An archive on the same disk is not an independent backup.

# Lossless authoring and validation

## Planning

The source manifest records original relative paths, byte sizes, and SHA-256 digests before edits. Each split segment identifies a half-open byte interval `[start, end)` and the exact payload offset in its generated document. Adjacent intervals must meet, start at zero, and end at the original size. Concatenation must reproduce the original bytes and digest. Even an empty original gets a zero-length record.

Choose boundaries by reading the content. Keep a code example with the assumptions and error discussion that make it intelligible. Split unrelated subjects into different concepts; link related mechanisms with a sentence explaining the relation. Retain duplicates in the source history. Merge their explanation only in the new reader-facing layer.

## Standalone enrichment

Explain each topic without mentioning a source filename or assuming the reader saw an earlier conversation. External citations support concrete claims. Cite versioned documentation for APIs. Separate a documented mechanism from a deduction, a recommended experiment, and an observed measurement. Replace grand claims about secret knowledge with precise details about invariants, failure windows, counterexamples, or ownership.

A maintenance coverage record can link original files to the new concepts. That record establishes traceability without forcing preservation material into the reader's explanation. Record full-content review honestly: indexing bytes, scanning headings, and rendering an excerpt are different from reviewing all of it.

## Validation interpretation

Default checks enforce the minimum OKF v0.2 structure. `--strict` also checks the local producer profile, optional fields, citation joins, and authored local file links. Custom types are accepted. Link fragments and remote availability are outside this static check. Attested computations are inspected, never executed.

Preservation failures are blocking for a lossless task. Do not update a manifest just to make a changed original pass. Identify the change and reconcile it with the user's intended source version. Inherited unresolved links are separate observations; new authored broken links must be fixed. A successful result is scoped to the validator implementation and its tested checks, not every possible semantic requirement in the specification.

## Agent handoff

The handoff includes the entry index, full command from any working directory, final report path, source count, preservation result, and actual research limitations. Include the script directory with the bundle. Never require a future agent to know this conversation or the original repository checkout to validate a copied bundle.

## Concurrent source revisions

When a file gains user edits after its initial fingerprint, keep the initial manifest and assets. Capture the newly reviewed revision as another asset and add exact source ranges for appended topics. The current manifest may select a bundle-relative `asset` on a file entry; omitted `asset` retains the default `assets/originals/<path>.txt` for Markdown. This is the preservation tool's optional extension, not an OKF metadata requirement. Absolute paths and parent traversal are rejected. Validate the initial manifest without `--originals` and the current manifest with `--originals` to distinguish historical snapshot integrity from current workspace equality. Never repair a mismatch by reverting user content or silently changing hashes without a reviewed snapshot.

`reindex_okf.py` also regenerates `maintenance/catalog.json` from current document frontmatter, so machine consumers do not retain a stale catalog after additions. Manual indexes remain manual; update their navigation deliberately.

## Read and audit before enrichment

For digital PDFs, start with local `pdftotext -layout` and inspect figures or tables whose meaning depends on layout. Empty or garbled extraction needs visual inspection and an appropriate local OCR path; do not treat successful process exit as complete extraction. Keep the complete PDF alongside extracted text.

For standalone-document audits, distinguish local source references from external URLs. An external documentation URL ending in `workflows.md` does not reference a local original merely because its basename matches. Check resolved local destinations and prose separately from external citations.

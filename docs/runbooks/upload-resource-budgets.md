# Upload and document-extraction budgets

The production defaults are deliberately bounded at two different edges:

| Boundary | Default | Enforcement | Why |
| --- | ---: | --- | --- |
| Multipart upload | 10 MiB (`10485760`) | `read_bounded_upload` reads at most `MAX_UPLOAD_SIZE + 1` bytes | A compact PDF can expand during OCR/table inference; rejecting the 10 MiB + 1 byte case prevents parser and object-store work. |
| Parser wall clock | 120 s | `asyncio.timeout` around the offloaded Docling conversion | OCR and table extraction are synchronous CPU work and must not hold an ingestion job indefinitely. |
| Parsed pages | 250 | Checked immediately after conversion | Page count is the most stable input-independent work bound for PDFs and scans. |
| Extracted output | 50 MiB UTF-8 | Checked before classification | Prevents pathological OCR expansion from flowing into classification, chunking, and persistence. |

## Benchmark evidence

Run the deterministic local benchmark with:

```bash
.venv/bin/python scripts/benchmark_upload_pipeline.py \
  --output evals/benchmarks/upload_pipeline_latest.json
```

The retained baseline uses `tools/lynk/testdata/contracts/minimal-valid.docx` (1,628 bytes,
non-confidential test data). On the 2026-10-27 local run, the real Docling parser and
classification boundary completed in 0.135 s, produced 388 UTF-8 bytes / an estimated 59 tokens,
and reported a 701,845,504-byte process peak (the process includes Docling and its OCR runtime;
the parse itself increased the observed peak by 1,048,576 bytes). Upload validation accepted 1 KiB,
1 MiB, 5 MiB and exactly 10 MiB; 10 MiB + 1 byte failed deterministically with
`FILE_TOO_LARGE` before extraction. The benchmark intentionally excludes databases, object
storage, embeddings, network access, and model-backed chunking so these measurements remain
repeatable and offline.

The 10 MiB limit is therefore retained rather than increased: the parser's process baseline is
already large, and an upload limit is a memory/DoS boundary, not merely a convenience limit. The
hard settings ceiling is 25 MiB; raising beyond that requires a new production-worker benchmark
and an explicit code/configuration change.

## Overrides and failure semantics

Operators can override `MAX_UPLOAD_SIZE`, `DOCUMENT_EXTRACTION_TIMEOUT_SECONDS`,
`DOCUMENT_EXTRACTION_MAX_PAGES`, and `DOCUMENT_EXTRACTION_MAX_OUTPUT_BYTES` through the normal
environment settings. Pydantic rejects non-positive values and values above the hard safety
ceilings (25 MiB, 900 s, 2,000 pages, and 250 MiB respectively). Rerun the benchmark against the
same container image after any override.

The timeout is a cancellation boundary for the async ingestion task. Docling runs in a worker
thread because it is synchronous; Python cannot safely kill that thread, so a timed-out conversion
may finish in the worker after the caller has received the budget failure. Deployments must bound
concurrent ingestion workers as well as individual uploads. Page and output checks happen inside
the worker and stop the pipeline before classification/persistence when exceeded.

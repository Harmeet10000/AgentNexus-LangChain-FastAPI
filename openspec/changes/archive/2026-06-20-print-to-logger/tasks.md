## 1. Audit & Prepare

- [ ] 1.1 Run `ast-grep run --pattern 'print($$$)' --lang python src/` and catalog all 30 print() calls by file
- [ ] 1.2 Identify which calls are user-facing output vs debug/progress vs error
- [ ] 1.3 Verify loguru is already imported in both affected files

## 2. rag_agent_advanced.py (17 print calls)

- [ ] 2.1 Replace print() calls in `main()` block (lines 510-580) with `logger.info()` / `logger.debug()`
- [ ] 2.2 Buffer streaming response and log full text via `logger.info()` after completion
- [ ] 2.3 Replace `print(f"\n\nError: {e}")` with `logger.error(f"...")`
- [ ] 2.4 Replace `print("\n\nGoodbye!")` and shutdown messages with `logger.info()`

## 3. ingest.py (13 print calls)

- [ ] 3.1 Replace progress print (line 512) with `logger.debug()`
- [ ] 3.2 Replace ingestion summary prints (lines 523-543) with `logger.info()`
- [ ] 3.3 Replace error prints with `logger.error()`
- [ ] 3.4 Replace interrupt message with `logger.info()`

## 4. Verify

- [ ] 4.1 Run `uv run ruff check src/app/shared/rag/rag_agent_advanced.py src/app/shared/rag/document_processing/ingest.py`
- [ ] 4.2 Run `uv run ruff format src/app/shared/rag/rag_agent_advanced.py src/app/shared/rag/document_processing/ingest.py`
- [ ] 4.3 Run `ast-grep run --pattern 'print($$$)' --lang python src/app/` and verify zero matches
- [ ] 4.4 Run `uv run pytest` to confirm no regressions

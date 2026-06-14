---
name: cognee-quickstart
description: Help users start or troubleshoot a Cognee project by choosing the right installation path, creating a clean Python environment, configuring LLM and embedding providers, installing optional extras, running the first remember/recall smoke test, and resolving dependency setup errors.
---

# Cognee Quickstart

Use this skill when the user is starting with Cognee, setting up dependencies, running Cognee from source, configuring providers, using local models, or reporting install/import/API key errors.

## First move

Classify the setup path before installing anything:

- **Fastest package setup**: User wants to try Cognee in a new app or notebook.
- **Provider-specific setup**: User wants OpenAI, Gemini, Anthropic, Ollama, Fastembed, or another backend.
- **Source setup**: User cloned `topoteretes/cognee` to inspect examples, change code, or contribute.
- **Existing broken setup**: User already has errors, stale environments, or dependency conflicts.

If the path is unclear, ask one short question. Otherwise make the conservative default: package install in a fresh virtual environment using OpenAI defaults.

## Environment rules

- Work from the project root, where the user's `.env` should live.
- Require Python 3.10 through 3.14. Check `python --version` or `python3 --version`.
- Prefer `uv` for environment and package work. Use `python -m venv` and `pip` only when `uv` is unavailable.
- Do not install packages globally unless the user explicitly asks.
- Do not mix package managers in the same environment.
- Treat `.env` values as secrets. Never print real API keys back to the user.

## Fastest package setup

Use this path for most first-time users with an OpenAI API key:

```bash
uv venv
source .venv/bin/activate
uv pip install cognee
```

Create `.env` in the project root:

```dotenv
LLM_API_KEY="your_openai_api_key"
```

Verify import before running a full example:

```bash
python -c "import cognee; print('cognee import ok')"
```

Then run the smoke test:

```python
import asyncio
import cognee

async def main():
    await cognee.forget(everything=True)
    await cognee.remember("Cognee turns documents into AI memory.")
    results = await cognee.recall(query_text="What does Cognee do?")
    for result in results:
        print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
```

## Provider setup

The most common configuration mistake is setting only an LLM provider or only an embedding provider. If the user is not using OpenAI defaults, configure both sides explicitly.

### Gemini example

```dotenv
LLM_PROVIDER="gemini"
LLM_MODEL="gemini/gemini-flash-latest"
LLM_API_KEY="your_gemini_api_key"

EMBEDDING_PROVIDER="gemini"
EMBEDDING_MODEL="gemini/gemini-embedding-001"
EMBEDDING_API_KEY="your_gemini_api_key"
```

### Anthropic LLM example

Anthropic is an LLM provider, not an embedding provider. Pair it with an embedding provider such as OpenAI, Gemini, or Fastembed.

```bash
uv pip install "cognee[anthropic]"
```

```dotenv
LLM_PROVIDER="anthropic"
LLM_MODEL="anthropic/<your_claude_model>"
LLM_API_KEY="your_anthropic_api_key"

EMBEDDING_PROVIDER="openai"
EMBEDDING_MODEL="openai/text-embedding-3-small"
EMBEDDING_API_KEY="your_openai_api_key"
```

### Local no-API-key example

Use Ollama for the LLM and Fastembed for CPU-friendly embeddings:

```bash
ollama pull llama3.1:8b
```

```dotenv
LLM_PROVIDER="ollama"
LLM_MODEL="llama3.1:8b"
LLM_ENDPOINT="http://localhost:11434/v1"
LLM_API_KEY="ollama"

EMBEDDING_PROVIDER="fastembed"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS="384"
```

If the user changes embedding providers or dimensions after a previous run, advise a local reset before re-running:

```python
import asyncio
import cognee

async def main():
    await cognee.prune.prune_system(metadata=True)

asyncio.run(main())
```

## Source setup

Use this path when the user cloned the Cognee repository:

```bash
git clone https://github.com/topoteretes/cognee.git
cd cognee
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

If the development extra fails because the package metadata changed, fall back to:

```bash
uv pip install -e .
```

Create `.env` from `.env.template` if it exists, then set at least `LLM_API_KEY` for the default OpenAI path. Run the same import check and smoke test before changing examples or source code.

## Dependency triage

When setup fails, do not reinstall blindly. Read the first meaningful error and match it:

| Error clue | Likely fix |
|---|---|
| `No module named cognee` | Activate the virtual environment and install `cognee` in that environment. |
| `Python 3.9`, `SyntaxError`, or resolver rejects Python | Switch to Python 3.10 through 3.14 and recreate the virtual environment. |
| `anthropic` | Install `cognee[anthropic]`. |
| `psycopg2`, `asyncpg`, or `pgvector` | Install `cognee[postgres]` or `cognee[postgres-binary]`. |
| `neo4j` | Install `cognee[neo4j]`. |
| `playwright`, `tavily`, or `beautifulsoup4` | Install `cognee[scraping]`. |
| `unstructured` | Install `cognee[docs]`. |
| `docling` | Install `cognee[docling]`. |
| `fastembed` | Install `cognee[fastembed]` or `cognee[codegraph]`. |
| `tree_sitter` | Install `cognee[codegraph]`. |
| `modal` | Install `cognee[distributed]`. |
| `redis` | Install `cognee[redis]`. |
| `baml` | Install `cognee[baml]`. |
| API key, auth, or provider fallback errors | Confirm `.env` is in the project root and both LLM and embedding settings are configured for non-default providers. |
| Embedding dimension mismatch or stale vector collections | Reset local metadata with `await cognee.prune.prune_system(metadata=True)` or use a new `SYSTEM_ROOT_DIRECTORY`. |
| LLM connection preflight times out on a local/small model | Add `COGNEE_SKIP_CONNECTION_TEST=true` to `.env`, then test with a small input. |

## Windows adjustments

Use PowerShell activation instead of `source`:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Use forward slashes or doubled backslashes in `.env` path values.

## Finish criteria

Before saying the setup is done, confirm:

- The intended Python interpreter is active.
- `import cognee` succeeds.
- `.env` is in the project root.
- LLM and embedding configuration are both explicit unless the user is using OpenAI defaults.
- The smoke test stores text and recalls an answer.
- Any installed extras match the feature the user is actually trying to use.

# LangChain FastAPI Production Template

A production-grade FastAPI application integrating LangChain, LangGraph, and LangSmith with Google's Gemini models, featuring Pinecone for vector storage, Docling for document processing, Crawl4AI for web scraping, and **MCP (Model Context Protocol)** for dynamic tool integration.

## 🚀 Features

### Core Framework

-   **LangChain Integration**: Complete integration with Google Gemini models for LLM operations
-   **LangGraph Workflows**: Graph-based reasoning and workflow management
-   **LangSmith Monitoring**: Comprehensive tracing, evaluation, and feedback loops

### Advanced Capabilities

-   **MCP Protocol**: Dynamic tool discovery and multi-server communication
-   **Vector Store**: Postgres's pg_vectorscale integration for efficient semantic search and RAG
-   **Document Processing**: Multi-format document parsing with Docling (PDF, DOCX, PPTX, HTML, Markdown)
-   **Web Crawling**: Intelligent web scraping with Crawl4AI (JavaScript rendering, rate limiting)
-   **Structured Outputs**: Type-safe LLM responses with Pydantic models
-   **Agent Workflows**: ReAct, Plan-and-Execute, and custom agent patterns
-   **Memory Management**: Persistent conversation history and checkpointing

### Production Features

-   **Production Ready**: Docker, monitoring, caching, and security best practices
-   **Async First**: Fully asynchronous architecture for high performance
-   **Type Safe**: Complete type hints and Pydantic validation
-   **Multi-Server Support**: Connect to multiple MCP servers simultaneously
-   **Caching**: Redis-based caching for improved performance
-   **Rate Limiting**: Built-in rate limiting and throttling
-   **Error Handling**: Comprehensive error handling and logging
-   **Observability**: LangSmith integration for tracing and monitoring

## 📋 Prerequisites

-   Python 3.12+
-   [uv](https://docs.astral.sh/uv/) - Fast Python package manager (recommended)
-   [ruff](https://docs.astral.sh/ruff/) - Fast Python linter anf formater (recommended)
-   [ty](https://docs.astral.sh/ty/) - Fast Python type checker (recommended)
-   Docker and Docker Compose
-   API Keys:
    -   Google Gemini API Key
    -   Pinecone API Key and Environment
    -   LangSmith API Key (optional)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harmeet10000/langchain-fastapi-production.git
cd langchain-fastapi-production
```

### 3. Set up environment variables

```bash
touch .env.development 
# Edit .env and add your API keys
```

### 4. Using Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f app
```

### 5. Local Development with uv (Recommended)

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies (reads pyproject.toml)
uv sync

# For dev dependencies too
uv sync --extra dev

# Run the application
uv run uvicorn src.app.main:app --reload --reload-dir src --host 0.0.0.0 --port 5000 --no-access-log
or
uv run python src/app/server.py

# Run Pre-commit hooks
uv run pre-commit run --all-files

# Create migration
uv run alembic revision --autogenerate -m "Add user table"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1

# Most common commands people actually type
uv run alembic revision --autogenerate -m        # make migrations
uv run alembic upgrade head                      # apply
uv run alembic current                           # check state
uv run alembic history --verbose                 # show history
uv run ruff check --fix                          # lint + auto-fix
uv run ruff format                               # format
uv run pytest -x                                 # test & stop on first failure
uv run celery -A celery_config worker --loglevel=info

```

## ⚡ Why Use uv?

`uv` is a fast Python package manager that offers significant advantages:

-   **10-100x faster** than pip for dependency resolution and installation
-   **Better dependency resolution** with fewer conflicts
-   **Built-in virtual environment management**
-   **Compatible with pip** and existing workflows
-   **Deterministic builds** with better lock file support
-   **Parallel downloads** and installations


## 📁 Project Structure

## 🔧 Configuration

## 🎯 Core Features Detail

### 1. LangChain Integration

-   **Chat Models**: Google Gemini Pro, Flash, and custom models
-   **Chains**: RAG, Conversation, Summarization, Q&A
-   **Tools**: Web search, file query, database queries, file operations
-   **Memory**: Conversation buffers, summaries, and entity tracking
-   **Callbacks**: Token counting, latency tracking, custom handlers

### 2. LangGraph Workflows

-   **State Management**: TypedDict-based state with checkpointing
-   **Conditional Routing**: Dynamic workflow paths based on state
-   **Human-in-the-Loop**: Approval gates and manual interventions
-   **Multi-Agent**: Orchestrate multiple specialized agents
-   **Streaming**: Real-time updates for long-running workflows

### 3. Vector Store & RAG

-   **Postgres Integration**: Production-grade vector storage
-   **Embeddings**: Google Vertex AI
-   **Chunking Strategies**: Recursive, semantic, and custom splitters
-   **Retrieval**: Similarity search, MMR, and hybrid search
-   **Re-ranking**: Cross-encoder and LLM-based re-ranking

### 4. Document Processing

-   **Supported Formats**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT
-   **OCR Support**: Extract text from scanned documents
-   **Metadata Extraction**: Automatic metadata detection
-   **Batch Processing**: Parallel document processing
-   **Storage**: MongoDB-based document store

### 5. Web Crawling

-   **JavaScript Rendering**: Playwright-based crawling
-   **Smart Extraction**: Automatic content detection
-   **Rate Limiting**: Respectful crawling with delays
-   **Link Following**: Recursive crawling with depth control
-   **Content Cleaning**: Remove ads, navigation, and boilerplate

### 6. MCP (Model Context Protocol)

-   **Multi-Server**: Connect to unlimited MCP servers
-   **Custom Servers**: Easy extension with custom tools
-   **Auto-Discovery**: Automatic tool detection and registration


## 📊 Monitoring

### LangSmith Integration

1. Set up LangSmith credentials in `.env`
2. Access traces at https://smith.langchain.com
3. Monitor:
    - Request traces
    - Token usage
    - Latency metrics
    - Error rates


## 🙏 Acknowledgments

-   LangChain team for the amazing framework and MCP adapters
-   Google for Gemini models
-   Anthropic for the Model Context Protocol specification
-   PsotgresSQL for vector database
-   FastAPI for the web framework
-   The open-source community


---

**Note**: This is a template project. Remember to:

1. **Install uv** for faster dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Add your API keys to `.env`
3. Install `FastMCP` for MCP support: `uv add fastmcp`
4. Configure MCP servers in `src/mcp/config/server_config.py`
5. Configure security settings for production
6. Set up proper monitoring and alerting
7. Review and adjust rate limits
8. Configure CORS for your domains
9. Test MCP servers before deploying to production
10. Use `uv lock` to generate lock files for reproducible builds

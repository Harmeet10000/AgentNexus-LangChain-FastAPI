# LangChain FastAPI Production Template

A production-grade FastAPI application integrating LangChain, LangGraph, and LangSmith with Google's Gemini models, featuring Pinecone for vector storage, Docling for document processing, Crawl4AI for web scraping, and **MCP (Model Context Protocol)** for dynamic tool integration.

## 🚀 Features

### Core Framework

-   **LangChain Integration**: Complete integration with Google Gemini models for LLM operations
-   **LangGraph Workflows**: Graph-based reasoning and workflow management
-   **LangSmith Monitoring**: Comprehensive tracing, evaluation, and feedback loops

### Advanced Capabilities

-   **MCP Protocol**: Dynamic tool discovery and multi-server communication
-   **Vector Store**: Pinecone integration for efficient semantic search and RAG
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
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 5000
```

## ⚡ Why Use uv?

`uv` is a fast Python package manager that offers significant advantages:

-   **10-100x faster** than pip for dependency resolution and installation
-   **Better dependency resolution** with fewer conflicts
-   **Built-in virtual environment management**
-   **Compatible with pip** and existing workflows
-   **Deterministic builds** with better lock file support
-   **Parallel downloads** and installations



## Middleware Execution Flow
```
Request Flow:
┌─────────────────────────────────────────────────────────────┐
│ 1. CORS Middleware (Preflight checks)                       │
│ 2. Trusted Host Middleware (Host validation)                │
│ 3. GZip Middleware (Compression)                            │
│ 4. Security Headers (Add security headers)                  │
│ 5. Correlation ID (Add tracking ID)                         │
│ 6. Metrics Middleware (Start timing)                        │
│ 7. Timeout Middleware (Wrap with timeout)                   │
│ 8. Error Handler (Catch exceptions)                         │
│ 9. Your Route Handler (/api/endpoint)                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
Response Flow (reverse order):
┌─────────────────────────────────────────────────────────────┐
│ 9. Route Handler Returns Response                           │
│ 8. Error Handler (Pass through or catch)                    │
│ 7. Timeout Middleware (Check timeout)                       │
│ 6. Metrics Middleware (Record duration)                     │
│ 5. Correlation ID (Add X-Correlation-ID header)             │
│ 4. Security Headers (Add headers to response)               │
│ 3. GZip Middleware (Compress if needed)                     │
│ 2. Trusted Host Middleware (Pass through)                   │
│ 1. CORS Middleware (Add CORS headers)                       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
my_fastapi_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   ├── shared/                           # Shared AI/ML components
│   │   ├── __init__.py
│   │   │
│   │   ├── langchain/                    # LangChain components
│   │   │   ├── __init__.py
│   │   │   ├── chains.py                 # Custom chains
│   │   │   ├── prompts.py                # Prompt templates
│   │   │   ├── agents.py                 # Agent configurations
│   │   │   ├── callbacks.py              # Custom callbacks
│   │   │   └── models.py                 # LLM model configurations
│   │   │
│   │   ├── langgraph/                    # LangGraph workflows
│   │   │   ├── __init__.py
│   │   │   ├── graphs.py                 # Graph definitions
│   │   │   ├── nodes.py                  # Custom nodes
│   │   │   ├── edges.py                  # Edge conditions
│   │   │   └── state.py                  # State management
│   │   │
│   │   ├── langsmith/                    # LangSmith integration
│   │   │   ├── __init__.py
│   │   │   ├── tracing.py                # Tracing configuration
│   │   │   ├── evaluation.py             # Evaluation sets
│   │   │   └── monitoring.py             # Performance monitoring
│   │   ├── agents/                       # Agent system
|   |   |   |
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py             # Base agent class
│   │   │   ├── agent_factory.py          # Agent creation factory
│   │   │   ├── agent_registry.py         # Agent registry
│   │   │   ├── memory/                   # Agent memory systems
│   │   │   │   ├── __init__.py
│   │   │   │   ├── conversation.py       # Conversation memory
│   │   │   │   ├── entity.py             # Entity memory
│   │   │   │   └── vector.py             # Vector memory
│   │   │   ├── tools/                    # Agent tools
│   │   │   │   ├── __init__.py
│   │   │   │   ├── search_tool.py
│   │   │   │   ├── calculator_tool.py
│   │   │   │   ├── code_executor_tool.py
│   │   │   │   └── database_tool.py
│   │   │   ├── types/                    # Predefined agent types
│   │   │   │   ├── __init__.py
│   │   │   │   ├── conversational.py     # Conversational agent
│   │   │   │   ├── research.py           # Research agent
│   │   │   │   ├── code_assistant.py     # Code assistant agent
│   │   │   │   └── data_analyst.py       # Data analyst agent
│   │   │   └── orchestration/            # Multi-agent orchestration
│   │   │       ├── __init__.py
│   │   │       ├── coordinator.py        # Agent coordinator
│   │   │       ├── communication.py      # Inter-agent communication
│   │   │       └── delegation.py         # Task delegation
│   │   │
│   │   ├── rag/                          # RAG components
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py              # Retrieval logic
│   │   │   ├── embeddings.py             # Embedding models
│   │   │   ├── reranker.py               # Reranking logic
│   │   │   ├── chunking.py               # Document chunking strategies
│   │   │   └── pipelines.py              # RAG pipelines
│   │   │
│   │   ├── vectorstore/                  # Vector database
│   │   │   ├── __init__.py
│   │   │   ├── pinecone_client.py        # Pinecone connection
│   │   │   ├── operations.py             # CRUD operations
│   │   │   ├── indexing.py               # Index management
│   │   │   └── search.py                 # Search strategies
│   │   │
│   │   ├── crawler/                      # Web crawling
│   │   │   ├── __init__.py
│   │   │   ├── crawl4ai_client.py        # Crawl4AI integration
│   │   │   ├── extractors.py             # Content extractors
│   │   │   ├── parsers.py                # HTML/content parsers
│   │   │   └── schedulers.py             # Crawl scheduling
│   │   │
│   │   ├── document_processing/          # Document handling
│   │   │   ├── __init__.py
│   │   │   ├── docling_client.py         # Docling integration
│   │   │   ├── loaders.py                # Document loaders
│   │   │   ├── converters.py             # Format converters
│   │   │   └── preprocessors.py          # Text preprocessing
│   │   │
│   │   └── utils/                        # Shared AI utilities
│   │       ├── __init__.py
│   │       ├── token_counter.py
│   │       ├── text_splitter.py
│   │       └── validators.py
│   │
│   ├── features/                         # Business features
│   │   ├── __init__.py
│   │   │
│   │   ├── chat/                         # AI Chat feature
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── schema.py
│   │   │   ├── router.py
│   │   │   ├── service.py                # Uses shared/langchain
│   │   │   ├── repository.py
│   │   │   ├── dependencies.py
│   │   │   └── constants.py
│   │   │
│   │   ├── documents/                    # Document management
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── schema.py
│   │   │   ├── router.py
│   │   │   ├── service.py                # Uses shared/document_processing
│   │   │   ├── repository.py
│   │   │   ├── dependencies.py
│   │   │   └── constants.py
│   │   │
│   │   ├── knowledge_base/               # RAG knowledge base
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── schema.py
│   │   │   ├── router.py
│   │   │   ├── service.py                # Uses shared/rag, shared/vectorstore
│   │   │   ├── repository.py
│   │   │   ├── dependencies.py
│   │   │   └── constants.py
│   │   │
│   │   ├── web_scraping/                 # Web scraping feature
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── schema.py
│   │   │   ├── router.py
│   │   │   ├── service.py                # Uses shared/crawler
│   │   │   ├── repository.py
│   │   │   ├── dependencies.py
│   │   │   └── constants.py
│   │   │
│   │   └── agents/                       # AI Agents feature
│   │       ├── __init__.py
│   │       ├── model.py
│   │       ├── schema.py
│   │       ├── router.py
│   │       ├── service.py                # Uses shared/langgraph
│   │       ├── repository.py
│   │       ├── dependencies.py
│   │       └── constants.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── router.py
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── error_handler.py
│   │   ├── request_logging.py
│   │   └── rate_limit.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── validators.py
│       ├── formatters.py
│       └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── shared/
│   │   │   ├── test_langchain.py
│   │   │   ├── test_rag.py
│   │   │   └── test_vectorstore.py
│   │   └── features/
│   │       ├── test_chat.py
│   │       └── test_knowledge_base.py
│   ├── integration/
│   │   └── test_api.py
│   └── e2e/
│       └── test_flows.py
│
├── alembic/
├── scripts/
│   ├── seed_data.py
│   ├── init_pinecone.py
│   └── index_documents.py
│
├── .env
├── .env.example
├── .gitignore
├── alembic.ini
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

## 🎯 Core Features Detail

### 1. LangChain Integration

-   **Chat Models**: Google Gemini Pro, Flash, and custom models
-   **Chains**: RAG, Conversation, Summarization, Q&A
-   **Tools**: Web search, calculations, database queries, file operations
-   **Memory**: Conversation buffers, summaries, and entity tracking
-   **Callbacks**: Token counting, latency tracking, custom handlers

### 2. LangGraph Workflows

-   **State Management**: TypedDict-based state with checkpointing
-   **Conditional Routing**: Dynamic workflow paths based on state
-   **Human-in-the-Loop**: Approval gates and manual interventions
-   **Multi-Agent**: Orchestrate multiple specialized agents
-   **Streaming**: Real-time updates for long-running workflows

### 3. Vector Store & RAG

-   **Pinecone Integration**: Production-grade vector storage
-   **Embeddings**: Google Vertex AI, OpenAI, and custom embeddings
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

## 📚 API Documentation

Once the application is running, you can access:

-   **Swagger UI**: http://localhost:5000/api/v1/docs
-   **ReDoc**: http://localhost:5000/api/v1/redoc
-   **OpenAPI JSON**: http://localhost:5000/api/v1/openapi.json

### Available Endpoints

1. **Chat** - `/api/v1/chat` - Conversational AI with Gemini
2. **RAG Query** - `/api/v1/rag/query` - Semantic search and retrieval
3. **MCP Agents** - `/api/v1/mcp-agents/execute` - Multi-tool agent execution
4. **Document Upload** - `/api/v1/documents/upload` - Multi-format document processing
5. **Web Crawling** - `/api/v1/crawl` - Intelligent web scraping
6. **Workflows** - `/api/v1/workflows/execute` - LangGraph workflow execution

## 📊 Monitoring

### LangSmith Integration

1. Set up LangSmith credentials in `.env`
2. Access traces at https://smith.langchain.com
3. Monitor:
    - Request traces
    - Token usage
    - Latency metrics
    - Error rates


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

-   LangChain team for the amazing framework and MCP adapters
-   Google for Gemini models
-   Anthropic for the Model Context Protocol specification
-   PsotgresSQL for vector database
-   FastAPI for the web framework
-   The open-source community

## 📮 Contact

For questions and support, please open an issue on GitHub.

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

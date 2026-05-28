# Retrieval RAG

## Retrieval Pipeline

Standard retrieval pipeline:

- sources
- document loaders
- documents
- chunks
- embeddings
- vector store
- retriever
- LLM answer

RAG architecture choices:

- 2-step RAG
- agentic RAG
- hybrid RAG

If you already have a strong existing knowledge base such as SQL, CRM, or internal docs, you do not necessarily need to rebuild it as a vector store. You can expose it as a tool and pass retrieved context to the LLM.

Add retrieval evaluation before production: retrieved documents, relevance, groundedness, and answer correctness.

## Embeddings

23. Embeddings are not cached by default. For deterministic embeddings, a simple cache can eliminate redundant calls.

71. Example embedding cache:

```python
import time
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore

underlying_embeddings = ...
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"First call took: {time.time() - tic:.2f} seconds")

tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"Second call took: {time.time() - tic:.2f} seconds")
```

## Document Loaders

69. All document loaders implement the `BaseLoader` interface.

Common API:

- `load()`
- `lazy_load()`

Docling loader notes:

- parses PDF, DOCX, PPTX, HTML, and other formats
- produces unified rich representations useful for RAG

## Text Splitting

70. Text structure-based splitting with `RecursiveCharacterTextSplitter`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)
```

Length-based splitting:

- token-based
- character-based

Example token-based splitter:

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
```

71. Document structure-based splitting is useful for:

- Markdown by headers
- HTML by tags
- JSON by object or array elements
- code by functions, classes, or logical blocks

## Vector Stores

72. Unified vector store interface:

- `add_documents`
- `delete`
- `similarity_search`

Initialization:

```python
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embedding=SomeEmbeddingModel())
```

Adding documents:

```python
vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
```

Deleting documents:

```python
vector_store.delete(ids=["id1"])
```

Similarity search:

```python
similar_docs = vector_store.similarity_search("your query here")
```

Metadata filtering:

```python
vector_store.similarity_search(
  "query",
  k=3,
  filter={"source": "tweets"}
)
```

Similarity metrics may use cosine similarity, Euclidean distance, or dot product. Efficient search may rely on indexing methods like HNSW depending on the store.

## Unassigned Note

73. Present in source but not mapped by the organizer. Preserved in the source document for future review.

# Re-ranking

## Resource
**Rerankers and Two-Stage Retrieval | Pinecone**
https://www.pinecone.io/learn/series/rag/rerankers/

## What It Is
Re-ranking uses a cross-encoder model to refine initial retrieval results by scoring query-document pairs more accurately. After a fast retriever (like vector search) returns candidate documents, the re-ranker evaluates each candidate with the query simultaneously, capturing richer semantic interactions. This two-stage approach balances speed and accuracy.

## Simple Example
```python
# Stage 1: Fast retrieval
candidates = vector_search(query, top_k=100)

# Stage 2: Re-rank with cross-encoder
reranker = CrossEncoder('ms-marco-MiniLM')
scored_results = []
for doc in candidates:
    score = reranker.predict([query, doc])
    scored_results.append((doc, score))

# Return top re-ranked results
final_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:10]
```

## Pros
Significantly improves retrieval precision by understanding query-document relationships. Works well as a refinement layer on top of existing systems.

## Cons
Computationally expensive compared to embedding models. Adds latency as each document must be processed with the query.

## When to Use It
Use when accuracy is more important than speed. Ideal for narrowing down a large candidate set to the most relevant documents.

## When NOT to Use It
Avoid when real-time performance is critical. Skip if you have limited compute resources or very large result sets to re-rank.


# Agentic RAG

## Resource
**What is Agentic RAG? Building Agents with Qdrant**
https://qdrant.tech/articles/agentic-rag/

## What It Is
Agentic RAG empowers autonomous agents with multiple tools to explore knowledge dynamically. Unlike traditional RAG's single vector search, agents can write SQL queries for structured data, perform web searches, read entire files, or query multiple vector stores based on query complexity. The agent reasons about which tools to use and in what order, adapting its strategy to the task.

## Simple Example
```python
# Agent decides which tools to use
agent = RAGAgent(tools=[vector_search, sql_query, web_search, file_reader])

query = "What were Q2 sales for ACME Corp?"

# Agent reasoning:
# 1. Checks if structured data needed → use SQL tool
result = agent.sql_tool("SELECT revenue FROM quarterly_sales WHERE company='ACME' AND quarter='Q2'")

# If insufficient, agent tries another tool
if not result.complete:
    result += agent.vector_search("ACME Q2 financial performance")
```

## Pros
Highly flexible and adapts to query complexity. Can access heterogeneous data sources (SQL, files, web, vectors) intelligently.

## Cons
Increased complexity and unpredictability in behavior. Higher latency and cost due to multi-step reasoning and tool calls.

## When to Use It
Use for complex queries requiring multiple data sources or exploration strategies. Ideal when you have diverse knowledge types (structured and unstructured).

## When NOT to Use It
Avoid for simple lookups where traditional RAG suffices. Skip when you need predictable, fast responses or have limited tool infrastructure.


# Knowledge Graphs

## Resource
**RAG Tutorial: How to Build a RAG System on a Knowledge Graph | Neo4j**
https://neo4j.com/blog/developer/rag-tutorial/

## What It Is
Knowledge Graph RAG (GraphRAG) combines vector search with graph databases to capture both semantic meaning and explicit relationships between entities. Instead of just retrieving similar text chunks, the system queries a graph of interconnected entities (nodes) and relationships (edges), providing structured, contextual information. This grounds LLM responses in factual relationships and prevents hallucinations.

## Simple Example
```python
# Knowledge graph structure
graph = {
    "ACME Corp": {
        "type": "Company",
        "relationships": {
            "HAS_CEO": "Jane Smith",
            "REPORTED_REVENUE": "$314M",
            "LOCATED_IN": "California"
        }
    }
}

# Query combining vector + graph
query = "Who runs ACME Corp?"

# 1. Vector search finds relevant entity
entity = vector_search(query)  # Returns "ACME Corp"

# 2. Traverse graph for relationships
result = graph.query(
    "MATCH (c:Company {name: 'ACME Corp'})-[:HAS_CEO]->(ceo) RETURN ceo"
)  # Returns "Jane Smith"

# 3. LLM generates answer with structured facts
answer = llm.generate(query, context=result)
```

## Pros
Captures explicit relationships that vectors miss. Reduces hallucinations by providing structured, factual connections.

## Cons
Requires building and maintaining a knowledge graph. Complex setup and querying compared to simple vector search.

## When to Use It
Use when relationships between entities are crucial to answers. Ideal for domains with complex interconnections (healthcare, finance, research).

## When NOT to Use It
Avoid when data lacks clear entities and relationships. Skip if you need rapid prototyping without graph infrastructure investment.


# Contextual Retrieval

## Resource
**Introducing Contextual Retrieval | Anthropic**
https://www.anthropic.com/news/contextual-retrieval

## What It Is
Contextual Retrieval, introduced by Anthropic, prepends chunk-specific explanatory context to each chunk before embedding and indexing. An LLM generates a brief description explaining what each chunk is about in relation to the entire document. This technique includes Contextual Embeddings and Contextual BM25, reducing retrieval failures by 49% alone and 67% when combined with re-ranking.

## Simple Example
```python
# Original chunk (lacks context)
chunk = "The company's revenue grew by 3% over the previous quarter."

# Generate contextual prefix with LLM
context = llm.generate(
    f"Document: {full_document}\n\nChunk: {chunk}\n\n"
    "Provide brief context for this chunk:"
)
# Returns: "This chunk is from ACME Corp's Q2 2023 SEC filing;
# previous quarter revenue was $314M."

# Embed with context
contextualized_chunk = context + " " + chunk
embedding = embed_model.encode(contextualized_chunk)

# Also create contextual BM25 index with the same contextualized chunks
bm25_index.add(contextualized_chunk)
```

## Pros
Dramatically improves retrieval accuracy by adding document context to chunks. Works with both vector embeddings and BM25 keyword search.

## Cons
Significantly increases indexing time and cost due to LLM calls for every chunk. Larger index size due to additional context text.

## When to Use It
Use when chunks lack standalone meaning without document context. Ideal for technical documents, financial reports, or dense reference materials.

## When NOT to Use It
Avoid when chunks are already self-contained and clear. Skip if indexing budget or time constraints are tight, or corpus updates frequently.



# Query Expansion

## Resource
**Advanced RAG: Query Expansion | Haystack**
https://haystack.deepset.ai/blog/query-expansion

## What It Is
Query Expansion enhances user queries by generating multiple variations or adding related terms before retrieval. An LLM automatically generates additional queries from different perspectives, capturing various aspects of the user's intent. This addresses vague or poorly formed queries and helps cover synonyms and similar meanings.

## Simple Example
```python
# Original query
user_query = "What is RAG?"

# LLM generates expanded queries
expanded_queries = [
    "What is Retrieval Augmented Generation?",
    "How does RAG work in AI systems?",
    "Explain RAG architecture and components"
]

# Retrieve documents for all queries
for query in expanded_queries:
    results = vector_search(query)
```

## Pros
Improves retrieval recall by capturing multiple interpretations of the query. Handles vague queries and terminology variations effectively.

## Cons
Increases latency and cost due to multiple LLM calls and retrievals. May introduce noise if expanded queries drift from original intent.

## When to Use It
Use when users provide short, ambiguous, or poorly-worded queries. Ideal for keyword-based retrieval systems that need semantic variations.

## When NOT to Use It
Avoid when queries are already specific and well-formed. Skip if latency is critical or when operating under strict cost constraints.


# Multi-Query RAG

## Resource
**Advanced RAG: Multi-Query Retriever Approach | Medium**
https://medium.com/@kbdhunga/advanced-rag-multi-query-retriever-approach-ad8cd0ea0f5b

## What It Is
Multi-Query RAG generates multiple reformulations of the original query, executes parallel searches, and aggregates results. An LLM creates diverse perspectives of the same question to overcome the limitations of distance-based retrieval. Results from all queries are combined (typically taking the unique union) to create a richer, more comprehensive result set.

## Simple Example
```python
# Generate multiple query perspectives
original_query = "How do I deploy a model?"

reformulated_queries = llm.generate([
    "What are model deployment steps?",
    "Best practices for deploying ML models",
    "Model deployment infrastructure options"
])

# Execute searches in parallel
all_results = []
for query in reformulated_queries:
    results = vector_search(query, top_k=20)
    all_results.extend(results)

# Deduplicate and return unique results
final_results = deduplicate(all_results)
```

## Pros
Mitigates single query bias and improves result diversity. Increases recall by capturing different interpretations of user intent.

## Cons
Higher computational cost due to multiple retrievals. May retrieve redundant or less relevant documents if queries overlap poorly.

## When to Use It
Use when user queries may have multiple valid interpretations. Ideal for improving recall on ambiguous or broad questions.

## When NOT to Use It
Avoid for very specific queries with clear intent. Skip when latency and cost are major constraints, or retrieval corpus is small.



# Context-Aware Chunking

## Resource
**Semantic Chunking for RAG | Medium**
https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

## What It Is
Context-Aware Chunking (also called semantic chunking) intelligently determines chunk boundaries based on semantic similarity rather than fixed sizes. It generates embeddings for sentences, compares their similarity, and groups semantically related content together. This ensures chunks contain coherent topics, improving embedding quality and retrieval accuracy.

## Simple Example
```python
# Split document into sentences
sentences = document.split_sentences()

# Generate embeddings for each sentence
embeddings = [embed_model.encode(s) for s in sentences]

# Calculate similarity between consecutive sentences
similarities = [cosine_similarity(embeddings[i], embeddings[i+1])
                for i in range(len(embeddings)-1)]

# Group sentences where similarity > threshold
chunks = []
current_chunk = [sentences[0]]
for i, sim in enumerate(similarities):
    if sim > 0.8:  # High similarity = same topic
        current_chunk.append(sentences[i+1])
    else:  # Low similarity = new topic
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i+1]]
```

## Pros
Creates semantically coherent chunks that improve embedding quality. Preserves topic continuity and contextual meaning within chunks.

## Cons
Computationally expensive due to embedding every sentence. Slower indexing process compared to simple fixed-size chunking.

## When to Use It
Use when document topics are diverse and intermingled. Ideal for complex documents where topic boundaries are important for retrieval.

## When NOT to Use It
Avoid when processing speed is critical or documents are already well-structured. Skip for homogeneous documents with consistent topics throughout.



# Late Chunking

## Resource
**Late Chunking in Long-Context Embedding Models | Jina AI**
https://jina.ai/news/late-chunking-in-long-context-embedding-models/

## What It Is
Late Chunking processes entire documents (or large sections) through the embedding model's transformer before splitting into chunks. Traditional "naive chunking" splits text first, losing long-distance context. Late Chunking embeds all tokens together, then applies chunking after the transformer but before pooling, preserving full contextual information in each chunk's embedding.

## Simple Example
```python
# Traditional chunking (loses context)
chunks = split_document(doc, chunk_size=512)
embeddings = [embed_model.encode(chunk) for chunk in chunks]

# Late Chunking (preserves context)
# 1. Process entire document through transformer
full_doc_embeddings = transformer_layer(doc)  # 8192 tokens max

# 2. Chunk the token embeddings (not the text)
chunk_boundaries = [0, 512, 1024, 1536, ...]
chunk_embeddings = []
for i in range(len(chunk_boundaries)-1):
    start, end = chunk_boundaries[i], chunk_boundaries[i+1]
    # Mean pool the token embeddings for this chunk
    chunk_emb = mean_pool(full_doc_embeddings[start:end])
    chunk_embeddings.append(chunk_emb)
```

## Pros
Maintains full document context in chunk embeddings, improving accuracy. Leverages long-context models (8K+ tokens) effectively.

## Cons
Requires long-context embedding models with high token limits. More complex implementation than standard chunking approaches.

## When to Use It
Use when document context is crucial for understanding chunks. Ideal for documents where meaning depends on long-distance relationships.

## When NOT to Use It
Avoid when using standard embedding models with small context windows. Skip if documents are already short and context is local.



# Hierarchical RAG

## Resource
**Document Hierarchy in RAG: Enhancing AI Efficiency | Medium**
https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9

## What It Is
Hierarchical RAG organizes documents in parent-child relationships, retrieving small chunks for accurate matching while providing larger parent contexts for generation. Child chunks are embedded and searched, but when a match is found, the system returns the parent chunk (containing broader context) to the LLM. Metadata maintains relationships between chunks, enabling efficient navigation of the hierarchy.

## Simple Example
```python
# Index structure
document = {
    "parent": "Q2 Financial Report - Full Section",
    "children": [
        "Revenue increased 3% to $314M",
        "Operating costs decreased 5%",
        "Net profit margin improved to 12%"
    ]
}

# Embed only child chunks
for child in document["children"]:
    index.add(embed(child), metadata={"parent_id": document["parent"]})

# Retrieval
query = "What was Q2 revenue?"
child_match = vector_search(query)  # Finds "Revenue increased 3%..."

# Return parent context instead of just the child
full_context = get_parent(child_match.metadata["parent_id"])
# LLM sees entire Q2 section for better reasoning
```

## Pros
Balances retrieval precision with generation context. Reduces noise in search while providing sufficient context for reasoning.

## Cons
Requires careful design of parent-child relationships. Adds complexity to indexing and retrieval logic.

## When to Use It
Use when small chunks match better but lack sufficient context for answers. Ideal for structured documents with natural hierarchies (sections, chapters).

## When NOT to Use It
Avoid when documents lack clear hierarchical structure. Skip if simple flat chunking provides adequate context for your use case.




# Self-Reflective RAG

## Resource
**Self-Reflective RAG with LangGraph | LangChain**
https://blog.langchain.com/agentic-rag-with-langgraph/

## What It Is
Self-Reflective RAG (including Self-RAG and Corrective RAG/CRAG) adds self-assessment and iterative refinement to retrieval. The system evaluates whether retrieved documents are relevant, grades response quality, and refines queries or retrieves additional information when necessary. It creates a feedback loop where the system critiques its own outputs and adapts until producing a satisfactory answer.

## Simple Example
```python
# Initial retrieval
query = "What is quantum computing?"
docs = vector_search(query)

# Self-reflection: Grade document relevance
grades = []
for doc in docs:
    grade = llm.evaluate(f"Is this document relevant to '{query}'? {doc}")
    grades.append(grade)

# If relevance is low, refine and retry
if avg(grades) < 0.7:
    refined_query = llm.refine(query, docs)
    docs = vector_search(refined_query)

# Generate answer
answer = llm.generate(query, docs)

# Self-reflection: Verify answer quality
if not llm.verify_answer(answer, docs):
    # Retrieve more context or refine further
    additional_docs = web_search(query)
    answer = llm.generate(query, docs + additional_docs)
```

## Pros
Improves answer quality through self-correction and validation. Adapts dynamically to poor retrieval results by refining approach.

## Cons
Significantly higher latency due to multiple LLM calls and iterations. Increased cost and complexity compared to single-pass RAG.

## When to Use It
Use when answer accuracy is critical and errors are costly. Ideal for complex queries where initial retrieval may be insufficient.

## When NOT to Use It
Avoid when real-time responses are required. Skip for simple queries or when operating under strict latency/cost budgets.



# Fine-tuned Embedding Models

## Resource
**Fine-tune Embedding models for Retrieval Augmented Generation (RAG) | Philipp Schmid**
https://www.philschmid.de/fine-tune-embedding-model-for-rag

## What It Is
Fine-tuning embedding models adapts pre-trained models to domain-specific data, improving retrieval accuracy for specialized vocabularies and contexts. Instead of using generic embeddings trained on broad data, you train the model on your specific corpus with relevant query-document pairs. This teaches the model to recognize domain jargon, relationships, and semantic patterns unique to your use case.

## Simple Example
```python
# Prepare domain-specific training data
training_data = [
    ("What is EBITDA?", "positive_doc_about_EBITDA.txt"),
    ("Explain capital expenditure", "capex_explanation.txt"),
    # ... thousands of query-document pairs
]

# Load pre-trained model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune on domain data
fine_tuned_model = base_model.fit(
    train_data=training_data,
    epochs=3,
    loss=MultipleNegativesRankingLoss()
)

# Use fine-tuned model for retrieval
query_embedding = fine_tuned_model.encode("What is working capital?")
# Better matches domain-specific financial documents
```

## Pros
Significantly improves retrieval accuracy for specialized domains (5-10% gains typical). Can achieve better performance with smaller model sizes after fine-tuning.

## Cons
Requires domain-specific training data (query-document pairs or synthetic data). Additional time and resources needed for training and evaluation.

## When to Use It
Use when working with specialized domains (medical, legal, technical) with unique terminology. Ideal when retrieval accuracy is suboptimal with generic embeddings.

## When NOT to Use It
Avoid when working with general knowledge or common topics. Skip if you lack training data or can't afford the fine-tuning investment.


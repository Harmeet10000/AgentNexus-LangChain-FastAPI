
# RAG & Tools

### Google Docs API gave better performance for converting docs to markdown than lamaparse, PdfPlumber, PyMuPDF

 pypdfium has the highest score for for matching docs/PDF parsing
metaData includes:
   source: filePath
   page_no: 0

custom_metadata includes:
   source: filePath
   page_no: 0
   document_summary:
   chunk_id:
   chunk_faqs:
   chunk_keywords:

1. Knowledge Processing (Chunking & Embeddings)
The Problem: Fixed-length chunking (e.g., 30 tokens) fragments context, destroying the interconnected nature of information (02:46).
The Solution: Use Semantic Chunking based on document structure (e.g., sections, headers) and Hierarchical Chunking to maintain parent-child relationships between text chunks (06:23).
Enhanced Embeddings: Move beyond single-vector embeddings to multi-vector embeddings (capturing embeddings at the token level) for richer semantic representation (12:23).
2. Query Understanding
The Problem: Naive similarity search often fails to understand the user's true intent, leading to irrelevant search results (15:39).
The Solution: Enhance queries with user context, meta-information, and entity extraction to identify the true intent, urgency, and relevant domain (19:45).
Knowledge Orchestration: Implement planning mechanisms to determine necessary permissions and data freshness requirements before retrieving information (22:09).
3. Hybrid Retrieval Systems
The Problem: relying solely on cosine similarity for retrieval is insufficient for complex queries (25:51).
The Solution: Implement a Hybrid Retrieval system that combines parallel searches across a Vector Store (semantic), Document Store (keyword/BM25/splade), and Graph Store (knowledge graph entities) (30:30).
Fusion Ranking: Combine results from these different methods using algorithms like Reciprocal Rank Fusion (RRF) to determine the best final chunks for the language model (33:04).

### Accuracy and reliability

## V1

You can evaluate how correct, truthful, and complete your agent’s answers are. For example:
Hallucination. Do responses contain facts or claims not present in the provided context? This is especially important for RAG applications.Faithfulness. Do responses accurately represent provided context?

Content similarity. Do responses maintain consistent information across diﬀerent phrasings?
Completeness. Do response includes all necessary information from the input or
context? Answer relevancy. How well do responses address the original query?

You can evaluate how well the model delivers its ﬁnal answer in line with requirements around format, style, clarity, and alignment.
Tone consistency. Do responses maintain the correct level of formality, technical complexity, emotional tone, and style?
Prompt Alignment. Do responses follow explicit instructions like length restrictions, required elements, and speciﬁc formatting requirements?
Summarization Quality. Do responses condense information accurately?
Consider eg information retention, factual accuracy, and conciseness?
Keyword Coverage. Does a response include technical terms and terminology use?

 Retrieval Metrics (1:32 - 2:48)

Context Precision: The fraction of retrieved chunks that are actually useful.
Context Recall: The percentage of relevant information from the entire corpus that was successfully found.
Mean Reciprocal Rank (MRR): Measures the placement of the most relevant chunk; early placement (rank 1) is critical since most systems only ingest the top few results.
4. Generation Metrics (2:48 - 3:49)

Faithfulness: A reference-free metric checking if every claim in the output is supported by the retrieved context.
Answer Relevance: A reference-free check to ensure the response answers the specific question asked.
Answer Correctness: A comparison against ground truth labels using semantic similarity.
5. Building Your Evaluation Test Set (3:49 - 4:47)

The foundation involves creating triplets: (Question, Answer, Relevant Chunk IDs).
Test sets must include edge cases such as ambiguous queries, multi-hop reasoning, and out-of-scope questions.
The test set is a living artifact that must be updated as user patterns evolve.
Faithfulness Check with LLM-as-Judge
Extract atomic claims (single indivisible facts) from the answer
Classify each claim as supported or unsupported by context
Score = supported claims / total claims
Production target: 0.85 to 0.95 depending on use case

Retrieval Metrics Calculation
Precision = true positives / total retrieved
Recall = true positives / total relevant in corpus
Reciprocal Rank = 1 / position of first relevant chunk
MRR = average reciprocal rank across test set

1. Step 2: Code Examples (4:47 - 7:04)

Faithfulness Check (4:52): Uses an LLM-as-a-Judge to extract atomic claims and classify them as supported or unsupported. Targets are typically 0.85–0.95.
Retrieval Metrics (6:06): Demonstrates how to programmatically calculate precision, recall, and reciprocal rank to identify performance bottlenecks.

1. Document Processing & Recursive Chunking (3:23 - 6:41)
This phase establishes the system's foundation. The speaker advocates for structure-aware recursive chunking over fixed-size splitting. By prioritizing document boundaries (chapters, sections, paragraphs), the system preserves semantic coherence, which is crucial for retrieval precision. He recommends a target of 300 to 800 tokens per chunk with a 100-token overlap to prevent information loss at boundaries.
2. Metadata & Embedding Strategy (6:42 - 12:47)
Metadata should be a first-class citizen (6:42). Storing source IDs, versions, and document types allows for pre-retrieval filtering, drastically reducing search space. For embeddings, the strategy is to batch API requests to minimize overhead, use persistent storage for embeddings, and track model versions to allow for selective re-embedding as models improve.
3. Vector Database Architecture (12:48 - 20:28)
For scale, the speaker discusses HNSW (Hierarchical Navigable Small World) indices to achieve logarithmic search complexity. At 10 million+ documents, he emphasizes:

Keeping the index memory-resident for millisecond latency.
Implementing date-based or type-based sharding to route queries efficiently.
Using primary-replica topologies to handle high read throughput.
4. Hybrid Retrieval & Re-ranking (20:29 - 27:09) To maximize recall, the system combines dense vector search (semantic) with sparse keyword search/BM25 (lexical). These are merged via Reciprocal Rank Fusion (RRF). Post-retrieval, a cross-encoder re-ranker is used to refine the top 100 candidates into the final top 10, typically improving precision by 20–30%.
5. LLM Integration & Context Assembly (27:10 - 35:15) Context assembly focuses on token budget management. Even with large context windows, the speaker suggests focusing on high-relevance chunks rather than raw volume to avoid diluting the LLM's performance. The prompt should explicitly require source citations to ensure trust and allow for graceful fallbacks when the context is insufficient.
6. Multi-Layer Caching & Cost Optimization (35:16 - 45:59) Caching is the most significant lever for cost reduction (70–80% for FAQ workloads). The architecture utilizes three layers:
Query Cache: For identical questions.
Embedding Cache: To avoid redundant API costs.
Retrieval Cache: For semantically similar queries. Proper TTL (Time-To-Live) and content-based invalidation patterns are required to maintain data freshness.
7. Operations & Evaluation (46:00 - 58:00) Production-grade RAG requires continuous evaluation using a golden dataset (100–500 diverse queries). The pipeline should measure precision, recall, and answer correctness automatically. Centralized monitoring for latencies and costs is mandatory for long-term stability and to prevent cascading failures via circuit breakers.

Vector Database Internals: HNSW, Sharding & Scaling (15:38 - 20:28)

The speaker explains that vector databases use HNSW (Hierarchical Navigable Small World) graphs to enable efficient approximate nearest neighbor search. Unlike brute-force linear search—which would require 10 million distance calculations for 10 million vectors—HNSW creates hierarchical structures that reduce this to a few thousand calculations, achieving logarithmic-like search complexity (16:05-16:45).

Key operational details include:

Memory Residency: To hit millisecond-level latency at the 95th percentile, the index structure must be in memory or memory-mapped with ample page cache (16:54-17:15).
Parameter Tuning: Users must balance recall, memory, and query speed by tuning parameters like the 'M' value (connections) and 'EF search.' The speaker warns against using default settings, which are designed for small demo datasets (17:23-18:05).
Sharding Strategies: For systems exceeding 10 million documents, sharding becomes necessary. Date-based sharding is highly effective for time-sensitive content: recent data is kept in 'hot' shards, while historical data resides in 'cold' shards, preventing the system from wasting cycles searching irrelevant history (18:30-19:35).
Scaling: The architecture supports linear scaling; doubling the shards roughly doubles the capacity, making capacity planning more predictable (20:17-20:25).The Retrieval Engine: Architecting Hybrid Search (20:29 - 24:42)
Hybrid search is presented as a method to combine the strengths of two distinct retrieval worlds:

Dense Vector Search: Excels at semantic understanding, matching intent even when terminology differs (20:48-20:56).
Sparse Keyword Search (BM25): Excels at exact matches, such as error codes or specific technical identifiers that might lose signal strength during embedding (21:00-21:12).
The implementation strategy involves running both searches in parallel (using tools like promise.all) to ensure the total latency is the maximum of the two search times rather than their sum (21:17-21:20). The results are then merged using Reciprocal Rank Fusion (RRF). RRF is preferred over score normalization because vector and keyword scores operate on different, non-comparable scales; RRF instead relies on the ranking position, which is more robust and requires no parameter tuning (21:21-23:32). After merging, the system performs deduplication by chunk ID before passing the final candidates to the re-ranker (23:33-23:40).Improving Precision: Re-ranking with Cross-Encoder Models (24:43 - 27:09)

Re-ranking is described as a critical refinement step that typically yields a 20-30% improvement in retrieval quality (24:47-24:52).

The speaker draws a sharp technical distinction between bi-encoder models (used for initial embedding generation) and cross-encoder models:

Bi-Encoders: Process queries and documents independently; they never 'see' the relationship between the two until the final dot-product or cosine similarity comparison (25:34-25:47).
Cross-Encoders: Process the query and document as a single, combined input. This allows the model to analyze the semantic relationship directly, making them far more accurate for assessing relevance (25:54-26:10).
While cross-encoders add 200-300 milliseconds of latency due to their higher computational cost, the speaker argues this is a necessary trade-off for production applications where quality directly dictates user satisfaction and reduces the need for expensive, secondary LLM calls (26:27-26:50).

Improving Precision: Re-ranking with Cross-Encoder Models (24:43 - 27:09)
Cross-Encoders vs. Bi-encoders: While bi-encoders (used for initial retrieval) process queries and documents independently, cross-encoders process them as a joint pair. This allows the model to deeply evaluate the relationship between the query and document, leading to significantly higher accuracy.
Precision Gains: Re-ranking the top 100 retrieval candidates typically improves precision by 20% to 30%.
Latency Trade-off: Cross-encoders are computationally expensive, adding roughly 200-300ms of latency. However, for applications where quality is paramount, this refinement stage is essential for filtering out noise.

Context Assembly & Token Budget Management (27:10 - 29:58)
Beyond Volume: More context does not always equate to better performance; excessive context can dilute the model's focus. The focus should be on relevance, not just filling the context window.
Assembly Strategy: Chunks should be assembled in descending order of relevance. Always include metadata to enable source attribution and citations.
Diversity & Fallbacks: To prevent redundancy, implement diversity constraints to avoid selecting similar chunks from the same document section. If retrieval scores are consistently low, implement a fallback strategy to avoid generating hallucinations.

LLM Integration & Production Prompt Engineering (29:59 - 35:15)
System Prompts: These are vital for defining behavioral constraints, such as forcing the model to answer only from the provided context and citing sources.
Streaming & Performance: Using streaming responses improves perceived latency, ensuring the user sees the answer being generated in real-time. Streaming also allows for early termination if a user cancels the query, which saves on token costs.
Reliability: Always implement exponential backoff and retries for LLM API calls, as service interruptions are common in production environments.
Performance & Cost: Multi-Layer Caching (35:16 - 42:22)

The Three-Layer Approach:
Query Cache: Stores final responses for identical user questions. This provides the fastest path to a response.
Embedding Cache: Stores query embeddings to avoid redundant API calls.
Retrieval Cache: A more advanced layer that caches results for semantically similar queries to skip the expensive vector search.
Cost Efficiency: In high-repetition environments (like FAQs), this approach can reduce operational costs by 70% to 80%.
Freshness: Use Time-To-Live (TTL) settings and content-based invalidation keys to ensure that cache entries are cleared when underlying documents are updated.

Cost Optimization Framework (42:23 - 45:59)
Economic Sustainability: Unoptimized RAG systems can become prohibitively expensive. The goal is to monitor cost-per-query as a primary KPI.
Optimization Tactics:
Model Routing: Direct simple queries to cheaper, smaller models, and reserve high-cost, advanced models for complex reasoning tasks.
Granular Monitoring: Track costs by individual components (embeddings vs. retrieval vs. generation) to identify where spend is concentrated.

Document Processing & Chunking
Recursive chunking strategies vs fixed-size splitting
Maintaining semantic coherence at scale
Metadata extraction and schema design

Vector Database Architecture
HNSW index internals and configuration
Sharding strategies for 10M+ documents
Primary-replica topology for high availability

Hybrid Search Implementation
Combining dense vector and sparse keyword retrieval
Reciprocal Rank Fusion for result merging
BM25 scoring and when keyword search matters

Re-ranking with Cross-Encoders
Bi-encoder vs cross-encoder architectures
Precision-latency trade-offs
Production model selection

Multi-Layer Caching
Query cache, embedding cache, retrieval cache
Cost reduction strategies (70-80% in FAQ workloads)
Cache invalidation patterns

## V2

 Advanced Document Chunking
Semantic chunking with embedding similarity detection
Hierarchical (parent-child) chunking for context preservation
Contextual chunking with LLM-generated preambles
Vision-based extraction for complex document formats

Query Understanding & Transformation
Query rewriting for vocabulary alignment
Query decomposition for multi-part questions
HyDE (Hypothetical Document Embeddings)
Building adaptive transformation pipelines

Agentic RAG Architectures
Router pattern for query classification
Tool-use and function calling for dynamic retrieval
Self-reflection and adaptive retrieval loops
Multi-hop reasoning across documents
Multi-agent orchestration patterns

Knowledge Graphs for RAG
Entity and relationship extraction
Hybrid graph + vector retrieval
When graph retrieval adds value vs. overhead

Multi-Modal RAG
Vision-language models for document understanding
Cross-modal embedding and retrieval (CLIP)
Complete multi-modal architecture design

Evaluation Deep-Dive
Component-level metrics (Recall@K, MRR, NDCG)
LLM-as-Judge patterns and limitations
Synthetic test set generation
Continuous online evaluation
Hallucination detection and faithfulness verification

Fine-Tuning for RAG
Embedding model fine-tuning with hard negatives
Reranker fine-tuning for domain precision
Generator fine-tuning for faithfulness

Production Hardening
Access control and multi-tenancy
Incremental indexing and cache invalidation
Guardrails and safety
Latency and cost optimization
Quality-focused monitoring

### advanced chunking strategies designed to overcome the limitations of standard approaches in production-grade RAG systems

1. Recursive Chunking Limitations (06:30 - 08:28)
While recursive chunking is a common starting point due to its simplicity and ability to respect natural text boundaries, it often fails in production because it optimizes for size rather than semantic coherence. Key failure modes include:

Context Loss: Separating a paragraph from its referenced table or a legal clause from its definition (e.g., spanning across multiple chunks).
Fragmented Logic: Splitting code examples mid-function or separating methodology descriptions from their key abbreviations.
Semantic Blindness: Because it relies on arbitrary size limits, it does not understand the underlying relationships between different sections of a document.

2. Semantic Chunking (08:28 - 10:23)
This approach uses embedding similarity to identify natural topic transitions rather than hard size limits.

Process: The document is split into sentences, which are then embedded. The system calculates the cosine similarity between consecutive sentences; sharp drops in similarity scores indicate topic boundaries.
Pros/Cons: It produces more coherent, self-contained chunks but leads to unpredictable chunk sizes and higher upfront computational costs for embedding every sentence.

3. Hierarchical (Parent-Child) Chunking (10:23 - 12:19)
This strategy balances the conflicting needs of retrieval (which prefers small, precise chunks) and generation (which requires broad, contextual chunks).

The Structure: A system stores small child chunks for efficient retrieval, each containing a pointer to a larger parent chunk (a section or full page).
The Workflow: Upon retrieving a child chunk, the system retrieves the full parent chunk to provide the LLM with the necessary surrounding context, effectively solving the "referenced table" problem mentioned earlier.

4. Contextual Chunking (12:19 - 14:04)
Contextual chunking uses an LLM to prepend a brief preamble to each chunk before embedding, explicitly situating that specific text within the larger document.

Value: A generic chunk like "revenue increased by 12%" is transformed into "This section discusses Q3 2024 financial results for the North American division: revenue increased by 12%." This ensures the embedding captures document-level context.

5. Handling Complex Document Formats (14:04 - 15:53)
Traditional text extraction often fails on enterprise documents like PDFs with multi-column layouts, tables, and charts.

Vision-based Extraction: The recommended solution is to pass pages as images to a vision-language model. This model performs implicit OCR, interprets the layout, and outputs structured data (e.g., Markdown tables) that preserve the integrity of the original format.

6. Chunking Strategy Decision Framework (15:53 - 17:46)
Mukul provides a helpful guide for choosing the right strategy based on document type:

**Narrative Documents** (Articles/Reports): Use Semantic Chunking combined with Hierarchical Chunking.
**Highly Structured Documents** (Legal/Specs): Use structure-aware Recursive Chunking with Contextual Preambles.
**Mixed Media** (Tables/Charts): Use Vision-based Extraction.
**General Knowledge Bases**: Simple recursive splitting is often sufficient for independent, well-defined articles.
**Pro-tip**: Always validate these strategies empirically by examining the resulting chunks and testing them against representative queries before finalizing your pipeline.

### raw user queries are often insufficient for high-performance RAG systems and outlines advanced transformation techniques to bridge the semantic gap

1. The Raw Query Problem (17:51 - 19:50)
The primary issue is that the user's natural language question is rarely the optimal search query. Challenges include:

Vocabulary Mismatch: Users may use colloquial terms (e.g., "firing employees") that do not align with formal document language (e.g., "termination procedures").
Implicit Context: Users often omit crucial details like jurisdiction or policy specifics, assuming the system knows the background.
Multi-part Questions: Complex queries requiring synthesis from multiple sources often fail if processed as a single embedding.
Ambiguous References: In a conversational flow, pronouns like "the other option" lose meaning without context.
2. Query Rewriting (19:50 - 21:33)
This technique uses an LLM to reformulate the user's input into an optimized retrieval query. It helps by:

Aligning Vocabulary: Automatically expanding terms with synonyms or domain-specific language.
Expanding Clarification: Using dialogue history to turn vague questions into specific ones (e.g., changing "What is the deadline?" to "What is the Q3 project deadline?").
Generating Multiple Interpretations: For ambiguous intents, generating several variations of the query can increase recall.
3. Query Decomposition (21:33 - 23:00)
For complex questions containing multiple parts (e.g., comparing 2024 pricing vs. 2023), a single pass is inadequate. Decomposition breaks these into:

Independent Subqueries: Each part is queried in parallel.
Unified Synthesis: The system retrieves chunks for each subquery and the LLM synthesizes them into one coherent answer.
4. HyDE: Hypothetical Document Embeddings (23:00 - 24:52)
HyDE addresses the asymmetry between interrogative questions and declarative documents. Instead of embedding the question, the system:

Generates a Hypothetical Answer: Uses an LLM to create a plausible (even if potentially imperfect) answer.
Embeds the Answer: Since the answer is declarative, it exists in the same semantic space as the source documents, leading to more accurate retrieval.
5. Query Transformation Pipeline (24:52 - 26:34)
Effective systems don't apply these techniques blindly; they use a classification step to determine the best approach:

Routing: Simple factoid queries pass through unchanged, while complex or ambiguous ones are routed to decomposition or rewriting.
Pipeline Order: Typically, decomposition occurs first, followed by rewriting specific subqueries.
Empirical Validation: It is essential to measure the impact of each transformation and disable those that degrade performance on specific query types.

### the transition from fixed, static RAG architectures to Agentic RAG, which uses LLMs to dynamically orchestrate retrieval and reasoning

1. From Static Pipelines to Adaptive Agents (26:39 - 28:11)
Static V1 pipelines process every query identically, regardless of complexity. Agentic RAG replaces this with an LLM-powered agent that operates in a reasoning loop:

Dynamic Decision Making: The agent decides what to retrieve, evaluates if the context is sufficient, and iterates if necessary.
Capability Shift: Unlike static systems, agents can reformulate queries based on initial results, synthesize information from multiple sources, and even decide when retrieval is unnecessary.
Trade-off: This introduces higher latency and cost due to multiple LLM calls per query.

2. The Router Pattern (28:11 - 29:31)
Routing is the simplest form of agentic RAG. A classifier or LLM analyzes incoming queries to direct them to the most effective retrieval backend:

Factoid Questions: Routed to vector search.
Analytical/Structured Queries: Routed to SQL databases.
Relational/Entity Queries: Routed to knowledge graph traversal.
Conversational/Other: Routed to dialogue history or answered directly without retrieval.

3. Tool Use for Retrieval (29:31 - 30:56)
By giving the LLM access to retrieval as a callable tool, the system becomes more flexible:

Definition: Tools are defined with specific capabilities (e.g., searching a specific index or fetching a document by ID) and instructions on when to use them.
Dynamic Parameters: The agent can set metadata filters or request more results if initial findings are insufficient, allowing the search strategy to change mid-reasoning.
Guardrails: It is critical to limit the number of tool calls per query and implement timeouts to avoid infinite, costly loops.

4. Self-Reflection and Adaptive Retrieval (30:56 - 32:15)
This pattern enables the agent to critique its own retrieved context before generating a final answer:

Judgment: The LLM acts as a judge, classifying the context as sufficient, insufficient, or contradictory.
Adaptive Loop: If the context is insufficient, the agent reformulates the query and tries again. If contradictory, it performs additional searches to resolve the conflict.

5. Multi-Hop Retrieval (32:15 - 33:41)
Some queries require chaining information across multiple documents (e.g., finding a policy in a memo, then finding the revenue data linked to that policy).

Sequential Reasoning: Unlike decomposition (which runs in parallel), multi-hop is inherently sequential because each step depends on the result of the previous one.
State Tracking: The agent must maintain state across hops to track identified entities and remaining unknowns.

6. Multi-Agent Architectures (33:41 - 34:49)
For complex enterprise systems, a planning agent can orchestrate specialized sub-agents:

Retrieval Agent: Focuses on search and context gathering.
SQL Agent: Focuses on database queries.
Summarization/Verification Agents: Handle condensing data or checking for hallucinations.
Recommendation: Start with a single-agent architecture (hub-and-spoke) and only graduate to multi-agent when query complexity demonstrably exceeds the capacity of a single agent.

7. Agentic RAG Decision Framework (34:49 - 36:12)
Use this framework to choose the right complexity level:

Static Pipeline: Use for uniform, low-complexity queries where latency and cost are the primary concerns.
Routing: Use when you have varied query types that can be mapped to predictable backends.
Single-Agent with Tools: Use for adaptive, multi-source retrieval requirements.
Multi-Agent: Reserved for the most complex systems where sub-tasks require specialized reasoning.

### Knowledge Graphs (KGs) in RAG systems, explaining how they complement vector search to handle complex relational queries

1. What Vector Search Cannot Do (36:14 - 37:34)
While vector search is excellent for finding semantically similar documents, it has fundamental limitations when it comes to structured relational data:

Lack of Intersection Logic: Vector search cannot perform set intersections required to find entities sharing multiple specific relationships (e.g., finding suppliers that contract with both Company A and Company B).
No Hierarchy Traversal: It cannot navigate organizational structures or citation networks, making questions about reporting lines or paper lineages impossible to answer directly.
Inability to Reason: Vector search lacks an understanding of entity properties and edge types, which is required for multi-step relational reasoning.

2. Building a Knowledge Graph (37:34 - 39:09)
Constructing a KG involves transforming raw text into a structured network of nodes and edges:

Entity Extraction: Identifying people, organizations, products, and domains using Named Entity Recognition (NER) or LLM-based extraction.
Relationship Extraction: Determining how entities interact (e.g., "John Smith is the CEO of Acme Corp" or "Acme Corp acquired Beta Co in 2023").
Graph Construction: Creating nodes for entities (with metadata) and edges for relationships (with types and temporal attributes).
Entity Resolution: A critical step to ensure that different names for the same entity (e.g., "John Smith" vs. "J. Smith") are merged into a single canonical identity.

3. Hybrid Graph + Vector Retrieval (39:09 - 40:17)
Optimal performance is achieved by combining the strengths of both methods:

Query Classification: The system determines if a user's prompt requires relational reasoning (KG) or content retrieval (vector).
Graph Traversal: The system executes a graph query to identify relevant entities and their connections.
Ranking Boost: The results from the graph traversal are used to provide a ranking boost to the relevant chunks retrieved by vector search. This ensures that the context provided to the LLM is both semantically relevant and relationally accurate.

4. Knowledge Graphs at Scale (40:17 - 41:31)
Scaling KGs requires managing complexity and costs effectively:

Incremental Extraction: Only process new or updated documents to avoid re-running expensive LLM extractions on the entire corpus.
Community Detection: Identify clusters of densely connected entities to create hierarchical summaries. This allows the system to answer "global" questions about entire topics by retrieving summaries instead of individual chunks.
Maintenance: Ensure the system handles updates and deletions, as changes in source documents must trigger updates to nodes and edges within the graph.

### Multi-Modal RAG how to extend RAG systems to process visual content, which is essential for enterprise documents containing tables, charts, diagrams, and scanned images

1. The Multi-Modal Challenge (41:33 - 42:56)
Standard text-only RAG pipelines often fail when encountering complex documents because they extract only text and discard visual information. This leads to:

Data Loss: Tables become scrambled, charts lose their labels, and diagrams disappear entirely.
Query Failure: Users asking about specific visual content (e.g., "What does the network topology show?") cannot be answered if the visual context was never preserved.

2. Vision Models for Document Understanding (42:56 - 44:15)
The recommended approach is to process document pages as images using Vision-Language Models (VLMs):

Process: At ingestion time, each page is sent to a VLM that extracts text, converts tables into Markdown, describes charts with trends and data points, and explains diagrams.
Benefits: This transforms visual content into structured text that is searchable through standard pipelines, capturing information that traditional OCR often misses.
Trade-off: This is a one-time, upfront cost during document ingestion, but it is necessary for high-quality retrieval.

3. Multi-Modal Embedding and Retrieval (44:15 - 45:32)
To enable cross-modal retrieval, the system uses models like CLIP to map images and text into the same shared vector space:

Embedding: Images are processed through a vision encoder to produce vectors, which are stored alongside text chunk vectors in the index.
Modality Filtering: At query time, the system can search across both text and image vectors. Metadata tags are used to filter results (e.g., searching only for images or only for text).
Limitation: Current multi-modal embeddings capture high-level semantics but may miss fine-grained visual details (like specific node labels in a diagram).

4. Multi-Modal Architecture (45:32 - 46:47)
A complete multi-modal architecture integrates these components:

Ingestion: Use format-appropriate extraction. Pages with complex layouts are sent to a VLM for structural extraction.
Retrieval: Depending on the user's query, the system retrieves either text chunks, images, or a combination of both.
Generation: The final response uses a Multi-Modal LLM that accepts both text and retrieved visual evidence, allowing it to synthesize answers that reference both types of data.

### Evaluating RAG systems is notoriously difficult because they are modular architectures where individual components can fail independently. The video highlights how end-to-end metrics often mask the root cause of a system failure, necessitating a shift toward component-level evaluation

1. Why Evaluation is Hard (46:51 - 48:06)
System failures are difficult to diagnose because an "incorrect" answer could stem from multiple points in the pipeline:

Retrieval failure: The query was misformulated or relevant content is missing.
Reranking failure: Good candidates were discarded by the reranker.
Generation failure: The model hallucinated or failed to synthesize provided context. End-to-end metrics tell you if the answer is bad, but component-level metrics are required to localize why it is bad.
2. Retrieval Evaluation Metrics (48:06 - 49:28)
To assess if the retriever is effectively surfacing relevant information, use these standard metrics:

Recall@K: The fraction of relevant documents found within the top K results.
Precision@K: The fraction of retrieved results that are actually relevant.
Mean Reciprocal Rank (MRR): Measures how high the first relevant document appears in the result list. Retrieval quality is the most critical diagnostic; if relevant information never reaches the generator, downstream improvements will not compensate.
3. Generation Evaluation (49:28 - 50:35)
This isolates the LLM's performance by providing it with "gold standard" context that definitely contains the answer. Key dimensions include:

Faithfulness: Does the answer rely only on the provided context?
Completeness: Does the answer cover all aspects of the query?
Relevance: Does the response address the user's intent?
Conciseness: Is the answer appropriately brief?
4. LLM-as-Judge (50:35 - 51:54)
Using an LLM to act as an automated grader is scalable but requires caution due to inherent biases:

Position/Verbosity Bias: Models may favor answers that appear first or are simply longer.
Self-Preference Bias: Models tend to favor outputs from their own family.
Calibration: It is essential to calibrate the judge by comparing its scores to human-annotated data and refining prompts accordingly.
5. Synthetic Test Set Generation (51:54 - 53:12)
To avoid the labor-intensive process of manual annotation, you can use LLMs to generate test sets:

Process: Sample chunks from your corpus and use an LLM to generate synthetic question-answer pairs.
Best Practices: Use different models for generation and evaluation to prevent circularity, and ensure your test set includes adversarial, edge-case questions rather than just "easy" factoid queries.
6. Continuous Online Evaluation (53:12 - 54:30)
Performance should be monitored post-deployment using both implicit and explicit signals:

Implicit: Look at query reformulation rates, conversation abandonment, and click-through rates on citations.
Explicit: Use user-provided thumbs up/down feedback.
Drift Detection: Longitudinal tracking helps identify if model quality is declining due to corpus changes or external data drifts.
7. Hallucination Detection (54:30 - 55:41)
To ensure groundedness, the system can perform claim verification:

Claim Decomposition: Break the generated response into atomic factual claims.
NLI (Natural Language Inference): Use models to classify each claim as supported, contradicted, or neutral relative to the retrieved context.
Mitigation: Post-generation filtering can block ungrounded answers, or the system can provide visual confidence indicators to the user.

### fine-tuning is a critical step for adapting general-purpose models to specific domain vocabularies, writing styles, and information patterns. It highlights three key components that can be fine-tuned to resolve specific bottlenecks in a RAG pipeline

1. Why Fine-Tuning Matters (55:43 - 56:41)
Off-the-shelf models are trained on general internet data. Fine-tuning allows your system to bridge the gap between this general knowledge and your organization's unique domain. The key is to start with evaluation to identify the specific failure point—whether it's poor retrieval recall, imprecise reranking, or subpar generation—and fine-tune only that component.

2. Fine-Tuning Embeddings (56:41 - 57:48)
Fine-tuning embedding models is used to improve retrieval recall.

Data: Uses query-passage pairs where the passage provides the answer. Data can be sourced from QA logs, manual annotations, or synthetic generation.
Hard Negative Mining: A crucial technique where you identify passages that are semantically similar but do not contain the answer. Training on these helps the model make finer distinctions.
Objective: Uses contrastive learning to pull positive pairs together while pushing negative ones apart. The video also mentions Matryoshka representation learning, which allows for embeddings that work well across multiple dimensions to trade off storage vs. quality.

3. Fine-Tuning Rerankers (57:48 - 58:32)
When retrieval returns relevant documents but they are ranked too low, fine-tuning the reranker acts as a precision filter.

Cross-Encoders: The model is trained to score query-document pairs jointly. This is often more effective than embedding fine-tuning for increasing precision.
Graded Relevance: Training often involves binary or graded labels (e.g., highly vs. somewhat relevant), teaching the model domain-specific relevance signals.

4. Fine-Tuning Generators (58:32 - 59:18)
When retrieved context is accurate but the generated response is poor, fine-tuning the generator is required.

Data Triples: Training uses (Question, Context, Answer) triples.
Negative Examples: It is essential to include examples where the context is insufficient, teaching the model to acknowledge uncertainty or abstain from answering rather than hallucinating.
Alignment: Proper citation training ensures the model explicitly links claims to supported chunks. Refusal training is also included to handle out-of-scope questions.

## V3

Context Strategy & Decision Framework
Long-context vs RAG trade-off analysis
Cost, latency, and precision comparisons
Hybrid retrieval + large context architectures
Decision framework for architecture choice

Multi-Turn Conversational RAG
Conversation memory and token budgeting
Query reformulation for follow-ups
Coreference resolution and entity tracking
Session state management with Redis
History compression strategies

Failure Diagnostics & Debugging
Retrieval vs generation failure taxonomy
Hallucination types and root causes
Lost-in-the-middle attention issues
Systematic diagnostic workflow
Building debugging tools

Edge Case Handling
No results and conflicting sources
Ambiguous and out-of-scope queries
Staleness detection and graceful degradation

Production Observability
Quality metrics beyond latency
Alerting and threshold calibration
Trace-based debugging

### technical analysis of the architectural trade-offs between Long-Context models and Retrieval-Augmented Generation (RAG) systems. Below is a detailed summary of the requested chapters

When Long-Context Wins Over RAG (04:09 - 06:55)
Long-context architectures are often superior when your entire dataset fits within the model's context window (e.g., up to 100k tokens).

Operational Simplicity: You avoid the engineering overhead of chunking, embedding management, and vector database maintenance.
Cross-Document Synthesis: Since the model has the entire corpus in view, it excels at global tasks like summarizing themes across multiple reports or identifying contradictions between documents—tasks where retrieval might fragment the necessary context.
Best for Small Teams: Startups with limited resources should prioritize the simplicity of long-context models over building complex RAG pipelines.
When RAG Remains Essential (06:55 - 09:21)
Despite the power of long-context, RAG is mandatory for production at scale:

Massive Corpora: If your data exceeds the context window (e.g., millions of records in healthcare or legal firms), RAG is your only option.
Continuous Freshness: If your documents update hourly, a RAG pipeline’s incremental indexing allows for real-time updates. Long-context approaches require re-uploading the entire dataset, which is operationally prohibitive.
Cost Analysis: Concrete Numbers (09:21 - 12:23)
Cost is often the primary driver for architectural choice.

The Math: Using GPT-4 pricing, a 200k token input costs roughly $2 per query. At 10,000 queries/day, that is $600k per month.
RAG Efficiency: By retrieving only relevant chunks (e.g., 5k tokens), your input cost drops significantly to pennies per query, leading to massive monthly savings even after accounting for vector database and embedding infrastructure.
Latency Comparison: Time to First Token (12:23 - 14:36)
Real-time constraints: If your application requires sub-second responses, long-context is typically non-viable because the model must process a massive prompt before generating the first token.
User Experience: RAG allows for faster latency by keeping the context prompt small and focused.
Precision Trade-offs: Needle in Haystack (14:36 - 17:03)
Attention Degradation: Research indicates models suffer from "lost-in-the-middle" phenomena, where they struggle to retrieve information buried in the center of long contexts.
Controlled Focus: RAG improves precision by pre-selecting the most relevant information and placing it in high-attention positions, effectively solving the "needle in the haystack" problem.
Hybrid Architectures: Best of Both Worlds (17:03 - 19:45)
Modern production systems combine the two. The pattern is:

Broad Retrieval: Use fast nearest-neighbor search to retrieve a larger set (20-50 chunks).
Reranking: Use cross-encoders to select the top 20 most relevant chunks.
Synthesis: Feed these chunks into the context window for the LLM to synthesize. This allows for deep reasoning over a focused, high-precision context.
Decision Framework: Choosing Your Approach (19:45)
Use these four filters to guide your architecture:

Corpus Size: <50k tokens = Long Context; >500k tokens = RAG.
Query Volume: >10,000 queries/day = RAG is mandatory for unit economics.
Latency: Sub-second requirement = RAG.
Update Frequency: Real-time/High frequency = RAG.

### the technical transition from simple retrieval systems to sophisticated, multi-turn conversational architectures. Below is a detailed breakdown of these core components

Implementation Patterns (22:12 - 25:06)
To build a robust production RAG system, engineers should follow these hierarchical and iterative patterns:

Hierarchical Retrieval: When a chunk is retrieved, include its parent section or document summary to provide framing and prevent context fragmentation.
Iterative Retrieval: For complex queries that cannot be answered in one pass, the system should generate an initial response, identify knowledge gaps, perform a second retrieval pass to fill those gaps, and then regenerate a final, more informed answer.
Single-Turn Limitations (25:06 - 27:28)
Single-turn systems, which treat every query as independent, are generally unsuitable for production chatbots because:

They cannot resolve pronouns (e.g., "that," "it") or positional references (e.g., "the second one") that rely on previous conversation history.
They lack "conversational memory," forcing users to re-state constraints or preferences, which creates a frustrating, unnatural user experience.
Conversation Memory Architectures (27:28 - 30:18)
To maintain continuity, systems must track the sequence of messages using:

Full History: Simple for short interactions but fails quickly due to token budget overflow as conversations lengthen.
Sliding Window: Retains only the last 5–10 turns. While this prevents overflow, it risks losing critical constraints established early in the chat.
Summarization: The most scalable approach; periodically compressing older history into a concise summary while keeping the most recent turns in high-fidelity detail.
Context Window Budgeting (30:18 - 32:47)
Proper budgeting is critical to avoid silent failures or API rejections. Developers should:

Establish a hard ceiling based on the model's limits.
Prioritize allocation: Reserve space for system prompts, conversation history, and an output buffer first; then allocate the remaining tokens to retrieved context.
Use dynamic budgeting to adjust retrieval depth (e.g., shorter conversations get richer context, while longer ones are forced to be more selective).
Query Reformulation (32:47 - 37:17)
Since vector databases cannot search for pronouns like "the second one," systems must use an LLM to rewrite user queries into standalone forms. This process uses recent conversation history to transform ambiguous statements (e.g., "What about that?") into searchable, context-rich queries (e.g., "Tell me the pricing for product B").

Coreference Resolution (37:17 - 40:35)
This is the process of mapping references to specific entities discussed previously. It requires:

Explicit Entity Tracking: Maintaining a structured list of products, people, or concepts mentioned in the chat.
Position-Based Handling: Specifically mapping ordinal references (e.g., "the first option") back to the exact list structure presented in the previous turn.
Re-Retrieval Strategy (40:35 - 43:04)
Retrieval should not always be a one-time event per turn. Systems should trigger re-retrieval when:

The conversation topic shifts significantly.
Retrieval confidence scores for the current context drop below an acceptable threshold.
The system detects that the user is attempting to clarify or dive deeper into a previously retrieved result, necessitating a fresh look at the corpus.

### the operational infrastructure required for production RAG systems, specifically focusing on state management, multi-turn handling, and diagnostic workflows

Session Architecture with Redis (43:04 - 45:30)
To support stateless application scaling, session data must be stored externally rather than in instance memory.

Redis Implementation: Use Redis for sub-millisecond session state management. History is stored as a list keyed by session ID.
Persistence: Use Rpush for adding new turns and Lrange for retrieving context, with TTL settings to automatically handle cleanup of inactive sessions.
Security: Strict key isolation is mandatory to prevent cross-user contamination.
History Compression Strategies (45:30 - 48:15)
To avoid token budget overflow in long conversations, implement tiered compression:

Recursive Summarization: Keep recent turns (e.g., the last 5) in high fidelity, while condensing middle turns into paragraphs and older turns into brief thematic notes.
Importance-Based Compression: Annotate turns with scores. Keep high-value turns containing critical constraints or decisions intact, while aggressively compressing low-importance social or clarifying turns.
Follow-Up Detection (48:15 - 50:18)
Efficient resource allocation depends on knowing when to trigger new retrieval:

Lexical Signals: Use keyword/pattern matching (e.g., "also," "what about," or pronouns) for cheap, initial classification.
Semantic Similarity: Compare query embeddings to recent history. Low similarity scores (e.g., < 0.4) strongly suggest a topic shift requiring a fresh retrieval.
Hybrid Approach: Use heuristics for clear cases and reserve LLM-based classification for ambiguous, complex queries.
Citation Handling Across Turns (50:18 - 52:24)
Maintaining trust requires accurate source attribution even when using old context:

State Tracking: Store retrieved document IDs and relevant snippets in your session state.
Consistency: Cite the original source document version retrieved at the time of the claim. Avoid vague attribution; provide specific document titles, section headers, or page numbers to facilitate user verification.
Conversation Boundaries (52:24 - 54:12)
Defining the scope of a conversation prevents context pollution:

Time-Based Expiration: Standardize breaks (e.g., 30 minutes for support, 24 hours for assistants).
Topic-Based Segmentation: Detect shifts in the conversation to segment context internally, ensuring that irrelevant topics do not interfere with retrieval accuracy for new subjects.
Failure Taxonomy: Retrieval vs Generation (54:12 - 56:35)
Effective debugging starts with correctly classifying the failure:

Retrieval Failure: The system fails to retrieve relevant documents because of embedding mismatches, poor chunking, or restrictive filters. Fixes involve improving embeddings, reranking, or metadata handling.
Generation Failure: The model ignores relevant retrieved context or hallucinates despite having the correct data. Fixes involve strengthening grounding instructions or adjusting the system prompt.
Retrieval Failure Patterns (56:35 - 58:50)
Embedding Mismatch: The user's vocabulary differs from the document's (e.g., "cancel subscription" vs "service termination"). Use query expansion or domain-specific fine-tuning.
Chunk Boundary Issues: Information is split across two chunks. Use overlapping windows or hierarchical chunking to ensure completeness.

### the critical task of diagnosing RAG failures, managing hallucinations, and building production-ready observability

Hallucination Types (58:50 - 1:01:05)
Effective debugging requires classifying the specific type of hallucination occurring:

Intrinsic Hallucination: The model directly contradicts retrieved context (e.g., claiming a 60-day refund window when the document says 30). This points to weak grounding or attention issues.
Extrinsic Hallucination: The model adds information not present in the context. This usually comes from the model's parametric knowledge and can be misleading when the user expects strictly grounded answers.
Fabricated Citations: The model invents sources or points to incorrect section numbers. This is particularly damaging to user trust.
Confidence Uncertainty: The model presents guesses as established facts. Ideally, the system should be calibrated to express uncertainty (e.g., "I don't see this in the documents") when appropriate.
Context Ignored: Lost in the Middle (1:01:05 - 1:03:12)
Models often exhibit "lost-in-the-middle" attention degradation, where information buried in the middle of a large prompt is ignored.

The Fix: Prioritize context ordering. Always place the highest-scoring retrieved chunks at the very beginning and the very end of the prompt to maximize the model's attention.
Instruction Reinforcement: Explicitly instruct the model to treat the provided context as the authoritative source and to quote passages directly when answering.
Diagnostic Workflow (1:03:12 - 1:05:40)
Never generalize a failure until you have diagnosed the specific instance:

Inspect Retrieval: Did the system retrieve relevant information? If not, stop—you have a retrieval problem (e.g., embedding mismatch, poor chunking).
Assess Ranking: Was the relevant document ranked too low (e.g., at rank 5 when it should be 1)? This suggests a need for better reranking.
Examine Utilization: If relevant docs were provided, did the model use them? If the model hallucinated anyway, it is a generation failure requiring better prompting or grounding.
Document findings: Maintain a pattern library of failures to inform future architecture improvements.
Building Debugging Tools (1:05:40 - 1:08:25)
Manual inspection does not scale. You must build internal tooling to:

Visualize retrieved chunks alongside the query and their associated relevance scores.
Isolate the generation step by feeding the exact retrieved context into the model again to see if it correctly synthesizes the answer.
Perform claim extraction, breaking responses into verifiable units to see which parts are grounded and which are hallucinated.
Root Cause Patterns (1:08:25 - 1:10:45)
Identifying recurring patterns helps prioritize engineering efforts.

Partial Correctness: Note that models often get some claims right while hallucinating others; segmenting these claims allows for more granular improvement.
Attribution Tracing: Automatically map model claims to source documents. If no link exists, flag the claim as a potential hallucination.
Prompt Sensitivity: If changing the prompt fixes a hallucination, the model was likely already capable of the answer but needed clearer instructions.
Edge Case Handling (1:10:45 - 1:14:20)
Handle the "known unknowns" gracefully:

No Results: When retrieval returns nothing, avoid forcing a hallucinated answer. Configure the model to state that the information is unavailable.
Conflicting Sources: If different documents contain contradictory information, the model should ideally present both perspectives rather than picking one arbitrarily.
Ambiguous Queries: Implement detection for out-of-scope or vague questions to avoid misleading the user.


### agent specific tool needs

Sources you must support:

Scanned PDFs (stamp papers, annexures)
Handwritten addendums
Poorly formatted Word/PDF files
Multi-language (English + Hindi + regional spillover)

Tools:

OCR (Indic language aware)
Layout-aware parsing (tables, schedules, annexures)
Metadata extraction (stamp duty, execution place, jurisdiction clause)

Data ingestion ≠ file upload
It includes:

Versioning (draft v/s executed v/s amended)
Annexure linking
Cross-reference resolution (“as per Clause 7.2(b)”)

If you skip this, lawyers won’t trust outputs.

B. Data Analysis Tools (beyond basic NLP)

NER alone is table stakes.

You need:

Entity normalization
(“Rs. 10 lakhs”, “₹10,00,000”, “Ten Lakh Rupees” → same value)
Temporal reasoning
“within 30 days of receipt” → receipt date + calendar + holidays
Conditional logic extraction
(“If X happens, Y obligation triggers”)
Jurisdictional mapping
Arbitration Act vs CPC
State stamp laws
Sectoral regulations

This is where LangGraph helps (multi-step reasoning).

C. Legal Knowledge Tools
Precedent linking (case law embeddings)
Statute grounding (section-level references)
Circulars / notifications (RBI, IRDAI, SEBI)

## Cost Optimization for RAG systems

GenAI Cost Fundamentals
Three primary cost domains: training, storage/retrieval, inference
Cost structure differences from traditional cloud computing
7B-13B parameter model economics (baseline for all estimates)

Fine-Tuning Economics
Upfront costs: $5K-$50K training compute for efficient methods (LoRA, QLoRA)
Monthly operating costs: $500-$3K per GPU for self-hosted serving
Break-even analysis: when $25K training investment pays off
Data requirements: 500-50K examples depending on task complexity
Parameter-efficient fine-tuning: 75-95% cost reduction vs full fine-tuning

RAG System Economics
Setup costs: $100-$1K for document ingestion and embedding
Monthly operating costs: $500-$5K vector database + query processing
Scaling behavior: linear cost growth with documents and queries
When RAG struggles: beyond 5M queries/month, beyond 50M documents
RAG advantages: zero training cost, faster time-to-production

Inference Optimization Techniques
Token optimization: prompt compression, context pruning, structured outputs
Caching strategies: 15-70% hit rates depending on query patterns
Cost reduction: 20-50% through semantic caching in production
Tiered model routing: 45% cost savings with complexity-based routing
Batch processing: 40-60% per-query savings (self-hosted only)

Cost Comparison & TCO Analysis
Monthly costs at 500K queries: $2.5K-$6K depending on architecture
Monthly costs at 5M queries: $8K-$60K depending on architecture
12-month TCO at different scales: $36K-$288K range
Break-even points: fine-tuning vs base models vs RAG
When fine-tuning achieves cost parity: 20-30M cumulative queries

Decision Framework
Query volume analysis: low (under 5M), medium (5M-20M), high (20M+) cumulative
Knowledge update frequency: static vs evolving vs hybrid
Quality requirements: high threshold vs good enough vs explainability needs
Hybrid architectures: combining fine-tuning + RAG for 35% cost savings

Production Implementation
Budget controls: per-team quotas, rate limiting, alerts
Cost monitoring: per-query metrics, token trends, cache hit rates
Reserved capacity strategy: 30-50% discounts with 60-70% baseline commitment
Continuous optimization: weekly reviews and iterative improvements

This detailed breakdown explores architectural patterns and cost-optimization strategies for GenAI, emphasizing that the best approach depends on scale, data volatility, and performance requirements.

Architectural Patterns

Architecture 1: Base Model + Optimizations (18:07)
Uses general-purpose foundation models with added caching and token optimization.
Pros: Fastest time to market and zero infrastructure maintenance (no training or retrieval pipelines).
Ideal for: Rapid prototyping, experimentation, or low-volume applications where the cost of custom development isn't justified.

Architecture 2: RAG (Retrieval-Augmented Generation) System + Optimizations (18:48)
Combines a foundation model with a vector database for external context.
Pros: Zero upfront training costs and excellent for applications requiring source attribution/citations.
Ideal for: Frequently changing knowledge bases where retraining would be prohibitive.

Architecture 3: Fine-tuned Model + Optimizations (19:28)
Involves high-upfront training costs, but results in a specialized, smaller, and more efficient model.
Pros: Lower per-query inference costs at scale due to model efficiency.
Ideal for: High cumulative query volume and stable, domain-specific requirements.

Cost Comparison & TCO
500,000 Queries/Month (20:18): Base models ($3K–$5K) and RAG ($3.5K–$6K) are generally cheaper than fine-tuning at this scale when considering the amortization of a $25K training investment ($2.5K–$4K monthly operational, excluding training costs).
5 Million Queries/Month (21:12): The economic advantage shifts. Base models scale linearly ($25K–$40K), while RAG becomes expensive due to retrieval overhead ($35K–$60K). Fine-tuning shines here ($8K–$18K) as the high upfront cost is diluted by high volume.
12-Month TCO Analysis (22:16): Fine-tuning achieves cost parity with other methods around 12–15 million cumulative queries. Beyond 24 million queries, fine-tuning can deliver 25–50% lower total costs compared to RAG or base models.

Decision Framework
Query Volume (24:55): <5M total queries favors base models; 5M–20M total queries is the "middle ground" for RAG; >20M total queries is the primary territory for fine-tuning.
Knowledge Update Frequency (25:46): Static knowledge favors fine-tuning (train once), while evolving knowledge requires RAG (re-embed documents rather than retrain).
Quality & Data Requirements (26:17): High-quality, specific task requirements often demand fine-tuning, provided you have sufficient training data (e.g., 500–50k examples). If explainability/citations are non-negotiable, RAG is the standard choice.

Hybrid Architectures (27:30)
Most high-performing production systems combine these approaches for efficiency:

Fine-tuning + RAG: Fine-tune for core domain language/jargon and use RAG for real-time data retrieval.
Tiered Routing + Caching: Use small, fast models for simple queries (via a router) and larger, expensive models only for complex tasks, while using semantic caching to eliminate redundant processing.
Example: A technical documentation system might use fine-tuning for code logic, RAG for API updates, and semantic caching for common developer questions, potentially reducing costs by 35% compared to a pure RAG

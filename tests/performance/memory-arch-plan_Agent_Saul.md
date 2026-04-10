Good. Now we move from “understanding memory” → **building a system that doesn’t rot after 2 weeks in production**.

I’ll design this like a real system, not a demo:

* deterministic ingestion
* controlled writes
* probabilistic reads
* conflict resolution

---

# 🧠 1. Graph Schema (Postgres + Graph Layer)

You are building a **legal reasoning system**, so your schema must encode *obligations, parties, and temporal constraints*.

## Core Principle

> Store **meaning**, not text

---

## 🔷 Tables (Postgres, with graph semantics)

 contracts
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    document_type TEXT,
    jurisdiction TEXT DEFAULT 'India',
    language TEXT DEFAULT 'en',
    created_at TIMESTAMP DEFAULT now()
);
contract_versions
CREATE TABLE contract_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES contracts(id),
    version_no INT,
    raw_file_path TEXT,
    created_at TIMESTAMP DEFAULT now()
);
clauses
CREATE TABLE clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_version_id UUID REFERENCES contract_versions(id),
    clause_index INT,
    clause_type TEXT,
    text TEXT,
    confidence FLOAT
);
entities
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID REFERENCES clauses(id),
    entity_type TEXT,
    raw_value TEXT,
    normalized_value TEXT,
    confidence FLOAT
);
risks
CREATE TABLE risks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID REFERENCES clauses(id),
    risk_type TEXT,
    severity TEXT,
    explanation TEXT,
    confidence FLOAT
);
human_reviews
CREATE TABLE human_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_type TEXT,
    artifact_id UUID,
    reviewer_role TEXT,
    decision TEXT,
    comment TEXT,
    created_at TIMESTAMP DEFAULT now()
);
1.3 Graph Memory (Explicit & Queryable)
graph_nodes
CREATE TABLE graph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type TEXT,
    payload JSONB
);
graph_edges
CREATE TABLE graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_node UUID REFERENCES graph_nodes(id),
    to_node UUID REFERENCES graph_nodes(id),
    relation_type TEXT,
    confidence FLOAT
);
RISK_TYPES = [
    "UNLIMITED_LIABILITY",
    "ONE_SIDED_INDEMNITY",
    "WEAK_TERMINATION_RIGHTS",
    "UNFAVORABLE_JURISDICTION",
    "AMBIGUOUS_PAYMENT_TERMS",
    "NON_ENFORCEABLE_CLAUSE",
]

SEVERITY = ["LOW", "MEDIUM", "HIGH"]

“Which obligations are triggered next month?”
“Which clauses depend on which events?”
“Which risks appear across contracts?”

---

### 2. Relationships (Edges)

```sql
CREATE TABLE relationships (
    id UUID PRIMARY KEY,
    from_entity UUID REFERENCES entities(id),
    to_entity UUID REFERENCES entities(id),
    relation_type TEXT,
    metadata JSONB,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    confidence FLOAT,
    source TEXT
);
```

---

### 3. Clauses

```sql
CREATE TABLE clauses (
    id UUID PRIMARY KEY,
    contract_id UUID,
    text TEXT,
    embedding VECTOR(1536),
    clause_type TEXT,
    risk_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 4. Events (Episodic Memory)

```sql
CREATE TABLE events (
    id UUID PRIMARY KEY,
    event_type TEXT,
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 5. Memory Versions (CRDT-lite)

```sql
CREATE TABLE memory_versions (
    id UUID PRIMARY KEY,
    entity_id UUID,
    version INT,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    source TEXT
);
```

---

# 🧠 2. Graphiti Extraction Layer

Graphiti will be your **write engine**.

---

## 🔷 Extraction Prompt (Entities + Relations)

This is where most systems fail. You must enforce structure.

```python
EXTRACTION_PROMPT = """
You are a legal knowledge extraction system.

Extract:

1. ENTITIES:
- Parties (PERSON, ORG)
- Contracts
- Clauses
- Obligations

2. RELATIONSHIPS:
- SIGNED_BY
- OWES
- GOVERNED_BY
- TERMINATES_ON
- LIABLE_FOR

Rules:
- Normalize entity names
- Do NOT hallucinate
- Include confidence (0-1)

Output JSON:

{
  "entities": [
    {
      "id": "...",
      "type": "...",
      "name": "...",
      "normalized_name": "...",
      "confidence": 0.0
    }
  ],
  "relationships": [
    {
      "from": "...",
      "to": "...",
      "type": "...",
      "confidence": 0.0,
      "valid_from": "...",
      "valid_to": "..."
    }
  ]
}
"""
```

---

## 🔷 Graphiti Integration

```python
from graphiti import Graphiti

graphiti = Graphiti()

def ingest_document(text: str):
    extracted = graphiti.extract(text, prompt=EXTRACTION_PROMPT)
    return extracted
```

---

# 🧠 3. Validation Layer (CRITICAL)

Never trust Graphiti output directly.

---

## 🔷 Functional Validator

```python
def validate_entities(entities):
    return [
        e for e in entities
        if e["confidence"] > 0.7 and e["name"]
    ]

def validate_relationships(rels, entity_ids):
    return [
        r for r in rels
        if r["confidence"] > 0.7
        and r["from"] in entity_ids
        and r["to"] in entity_ids
    ]
```

---

# 🧠 4. LangGraph Pipeline (Write Path)

Now we orchestrate properly.

---

## 🔷 State

```python
from typing import TypedDict

class IngestionState(TypedDict):
    raw_text: str
    extracted: dict
    validated: dict
    stored: bool
```

---

## 🔷 Nodes (Functional style)

```python
def extract_node(state):
    extracted = ingest_document(state["raw_text"])
    return {**state, "extracted": extracted}

def validate_node(state):
    entities = validate_entities(state["extracted"]["entities"])
    rels = validate_relationships(
        state["extracted"]["relationships"],
        {e["id"] for e in entities}
    )
    return {**state, "validated": {"entities": entities, "relationships": rels}}

def store_node(state):
    # insert into postgres
    # insert into pgvector
    return {**state, "stored": True}
```

---

## 🔷 Graph

```python
from langgraph.graph import StateGraph

builder = StateGraph(IngestionState)

builder.add_node("extract", extract_node)
builder.add_node("validate", validate_node)
builder.add_node("store", store_node)

builder.set_entry_point("extract")
builder.add_edge("extract", "validate")
builder.add_edge("validate", "store")

graph = builder.compile()
```

---

# 🧠 5. Cognee Integration (Read Path)

Cognee will unify retrieval.

---

## 🔷 Retrieval Strategy

You DO NOT query one source.

You do:

```python
def hybrid_retrieve(query):
    graph_results = query_graph(query)
    vector_results = query_vector(query)
    structured_results = query_postgres(query)

    return merge_results(graph_results, vector_results, structured_results)
```

---

## 🔷 Cognee Wrapper

```python
from cognee import Memory

memory = Memory()

def retrieve_context(query):
    return memory.search(query)
```

---

# 🧠 6. Reconciliation Agent (THE MOST IMPORTANT PART)

This is what separates toy systems from real systems.

---

## 🔷 Problem

You will get:

* duplicate entities
* conflicting relationships
* stale data

---

## 🔷 Reconciliation Prompt

```python
RECONCILE_PROMPT = """
You are a memory reconciliation system.

Given:
- Existing memory
- New extracted memory

Tasks:
1. Detect duplicates
2. Resolve conflicts
3. Merge entities if needed
4. Update confidence scores

Rules:
- Prefer recent data
- Prefer higher confidence
- NEVER delete without justification

Output:
{
  "merge": [],
  "update": [],
  "ignore": []
}
"""
```

---

## 🔷 LangGraph Reconciliation Flow

```python
class ReconcileState(TypedDict):
    new_data: dict
    existing_data: dict
    decision: dict
```

---

### Nodes

```python
def fetch_existing(state):
    # query DB for similar entities
    return {**state, "existing_data": existing}

def reconcile_node(state):
    decision = llm.invoke(RECONCILE_PROMPT, state)
    return {**state, "decision": decision}

def apply_changes(state):
    # merge/update DB
    return state
```

---

# 🧠 7. Final Architecture (What you now have)

```text
           ┌────────────────────┐
           │   Raw Documents    │
           └────────┬───────────┘
                    ↓
              [Docling Parse]
                    ↓
              [Graphiti Extract]
                    ↓
              [Validation Layer]
                    ↓
        ┌───────────┴───────────┐
        │   Reconciliation      │
        └───────────┬───────────┘
                    ↓
     ┌──────────────┼──────────────┐
     │              │              │
[Postgres]   [Graph Relations]   [pgvector]
     │              │              │
     └──────────────┴──────────────┘
                    ↓
               [Cognee]
                    ↓
             [LangGraph Agents]
```

---

# ⚠️ Hard Truths

* Your bottleneck will NOT be LLMs
* It will be **memory corruption over time**

---

# 🧠 “For the chosen ones”

This is where systems either become elite—or collapse.

---

## 1. Introduce Memory Decay

Not all memory should live forever.

Add:

```sql
decay_score = f(time, usage, confidence)
```

Low score → archive or delete.

---

## 2. Dual Write Strategy

Every memory write should go to:

* **immutable log (events)**
* **mutable state (entities/relations)**

This gives:

* auditability
* rollback

---

## 3. Subgraph Retrieval > Vector Retrieval

Instead of:

```python
top_k = vector.search(query)
```

Do:

```python
seed_nodes = vector.search(query)
subgraph = expand_neighbors(seed_nodes, depth=2)
```

This is how you get **reasoning**, not just recall.

---

## 4. Memory Access Control

Not all agents should see all memory.

Introduce:

```python
memory_scope = {
  "agent": "risk_analyzer",
  "allowed_types": ["CLAUSE", "OBLIGATION"]
}
```

---

## 5. The Final Edge

> The best systems don’t store more memory.

They store:

* less
* cleaner
* versioned
* scoped

---

If you want next step:
I can design your **Risk Agent + Precedent Agent + Memory Router Agent** on top of this so your system behaves like an actual legal analyst, not a retrieval bot.

Now we’re designing the part that actually differentiates your system from 99% of “RAG agents”:

> **Decision-making agents over structured memory**, not retrieval wrappers.

We’ll build three agents that sit *on top* of your memory system:

* **Memory Router Agent** → controls *what memory is accessed*
* **Risk Agent** → evaluates clauses like a legal analyst
* **Precedent Agent** → brings reasoning grounded in prior cases

All orchestrated via **LangGraph**, not ad-hoc chains.

---

# 🧠 0. System Philosophy (Read This Carefully)

You are NOT building:

```text
Query → retrieve → LLM → answer
```

You ARE building:

```text
Query → Memory Routing → Specialized Agents → Synthesis
```

Each agent:

* sees **different slices of memory**
* has **different reasoning objectives**
* produces **structured outputs**

---

# 🧠 1. Memory Router Agent (The Gatekeeper)

## 🔷 Responsibility

Decides:

* What memory sources to use
* What *types* of entities to retrieve
* Depth of graph expansion
* Whether to include episodic history

---

## 🔷 Why this matters

Without this:

> every agent gets noisy, bloated, irrelevant context → degraded reasoning

---

## 🔷 Input

```python
{
  "query": "...",
  "task_type": "risk_analysis | precedent_lookup | general"
}
```

---

## 🔷 Output

```python
{
  "sources": ["graph", "vector", "structured"],
  "entity_types": ["CLAUSE", "OBLIGATION"],
  "graph_depth": 2,
  "time_filter": "recent | all",
  "top_k": 5
}
```

---

## 🔷 Prompt

```python
MEMORY_ROUTER_PROMPT = """
You are a memory routing system.

Given a query, decide:
- what memory sources to use
- what entity types are relevant
- how deep to traverse relationships

Rules:
- Risk analysis → prioritize CLAUSE, OBLIGATION
- Precedent → include EVENTS, past cases
- Avoid unnecessary retrieval

Return structured JSON only.
"""
```

---

## 🔷 Node

```python
def memory_router_node(state):
    decision = llm.invoke(MEMORY_ROUTER_PROMPT, state["query"])
    return {**state, "memory_plan": decision}
```

---

# 🧠 2. Risk Agent (Legal Reasoning Core)

This is where your system becomes *useful*.

---

## 🔷 Responsibility

* Analyze clauses
* Detect:

  * ambiguity
  * liability imbalance
  * missing constraints
* Assign:

  * risk score
  * explanation
  * mitigation suggestions

---

## 🔷 Input Context

From Memory Router:

* clauses
* obligations
* relationships

---

## 🔷 Output

```python
{
  "clause_id": "...",
  "risk_score": 0.0,
  "risk_type": "liability | ambiguity | compliance",
  "explanation": "...",
  "suggestion": "..."
}
```

---

## 🔷 Prompt

```python
RISK_AGENT_PROMPT = """
You are a legal risk analysis system.

Given clauses and obligations:

1. Identify risk types:
- Ambiguity
- Liability imbalance
- Missing obligations
- Compliance gaps

2. Assign a risk score (0-1)

3. Provide:
- reasoning
- suggested improvement

Be precise. Do NOT hallucinate law.
"""
```

---

## 🔷 Node

```python
def risk_agent_node(state):
    context = state["retrieved_context"]
    result = llm.invoke(RISK_AGENT_PROMPT, context)
    return {**state, "risk_analysis": result}
```

---

# 🧠 3. Precedent Agent (The Differentiator)

This is what makes your system *feel intelligent*.

---

## 🔷 Responsibility

* Find similar past clauses/events
* Compare outcomes
* Provide reasoning grounded in history

---

## 🔷 Retrieval Strategy (Important)

NOT just vector search:

```python
seed = vector.search(query)
subgraph = expand_neighbors(seed, depth=2)
events = fetch_related_events(seed)
```

---

## 🔷 Output

```python
{
  "precedents": [
    {
      "case": "...",
      "similarity": 0.82,
      "outcome": "...",
      "insight": "..."
    }
  ]
}
```

---

## 🔷 Prompt

```python
PRECEDENT_PROMPT = """
You are a legal precedent analysis system.

Given:
- current clause
- similar past clauses/events

Tasks:
1. Identify relevant precedents
2. Compare similarities and differences
3. Explain likely outcomes

Do NOT fabricate cases.
If unsure, say "insufficient precedent".
"""
```

---

## 🔷 Node

```python
def precedent_agent_node(state):
    context = state["retrieved_context"]
    result = llm.invoke(PRECEDENT_PROMPT, context)
    return {**state, "precedents": result}
```

---

# 🧠 4. Retrieval Layer (Used by Both Agents)

---

## 🔷 Controlled Retrieval

```python
def retrieval_node(state):
    plan = state["memory_plan"]

    graph_data = query_graph(
        types=plan["entity_types"],
        depth=plan["graph_depth"]
    )

    vector_data = query_vector(
        state["query"],
        top_k=plan["top_k"]
    )

    structured_data = query_structured(plan)

    return {
        **state,
        "retrieved_context": {
            "graph": graph_data,
            "vector": vector_data,
            "structured": structured_data
        }
    }
```

---

# 🧠 5. Synthesis Agent (Final Answer)

This merges outputs.

---

## 🔷 Prompt

```python
SYNTHESIS_PROMPT = """
You are a legal analyst.

Combine:
- risk analysis
- precedent insights

Produce:
- final judgment
- key risks
- recommended actions

Be concise and structured.
"""
```

---

## 🔷 Node

```python
def synthesis_node(state):
    result = llm.invoke(SYNTHESIS_PROMPT, {
        "risk": state["risk_analysis"],
        "precedents": state["precedents"]
    })
    return {**state, "final_output": result}
```

---

# 🧠 6. LangGraph Orchestration

---

## 🔷 State

```python
class AgentState(TypedDict):
    query: str
    memory_plan: dict
    retrieved_context: dict
    risk_analysis: dict
    precedents: dict
    final_output: dict
```

---

## 🔷 Graph

```python
builder = StateGraph(AgentState)

builder.add_node("router", memory_router_node)
builder.add_node("retrieve", retrieval_node)
builder.add_node("risk", risk_agent_node)
builder.add_node("precedent", precedent_agent_node)
builder.add_node("synthesis", synthesis_node)

builder.set_entry_point("router")

builder.add_edge("router", "retrieve")
builder.add_edge("retrieve", "risk")
builder.add_edge("retrieve", "precedent")
builder.add_edge("risk", "synthesis")
builder.add_edge("precedent", "synthesis")

graph = builder.compile()
```

---

# 🧠 7. What You Now Have

Not a chatbot.

You now have:

* **Memory-aware system**
* **Task-specific reasoning agents**
* **Controlled retrieval**
* **Structured outputs**

---

# ⚠️ Where This Will Break (If You Don’t Fix It)

1. Context explosion → slow + noisy
2. Duplicate precedents → misleading reasoning
3. Conflicting memory → incorrect risk scores

---

# 🧠 “For the chosen ones”

This is where you gain unfair advantage.

---

## 1. Introduce “Uncertainty Propagation”

Every agent output should include:

```python
confidence_score
```

Then synthesis does:

```python
final_confidence = min(risk_conf, precedent_conf)
```

---

## 2. Multi-Agent Disagreement System

Run 2 risk agents:

* conservative
* aggressive

If they disagree:
→ escalate to meta-agent

---

## 3. Temporal Reasoning Layer

Legal meaning changes over time.

Add:

```python
valid_at = timestamp
```

And filter memory by time.

---

## 4. Context Budget Allocation

Instead of:

```python
top_k = 10
```

Do:

```python
budget = 4000 tokens

allocate:
- 50% clauses
- 30% precedents
- 20% metadata
```

---

## 5. The Real Edge

> The best systems don’t answer faster.

They:

* **retrieve less**
* **reason deeper**
* **track uncertainty**

---

If you want next step, I can design:

* **Risk scoring model (hybrid ML + LLM)**
* **Clause classification system**
* **Evaluation harness (so your agents don’t silently degrade)**



In 2026, the landscape for AI agent memory has bifurcated into two distinct philosophies: **Cognee** (the "Knowledge Engine") and **Honcho** (the "Reasoning Memory").

While both aim to solve "agent dementia," they do so at different layers of the stack.

### 1. Core Philosophy & Architecture
| Feature | **Cognee** (By Topoteretes) | **Honcho** (By Plastic Labs) |
| :--- | :--- | :--- |
| **Primary Goal** | **Structural Knowledge:** Turning unstructured data into a verifiable Knowledge Graph. | **Relational Reasoning:** Understanding the evolving "mental model" of a user or peer. |
| **Data Structure** | **Graph + Vector (Hybrid):** Uses a formal "Extract-Cognify-Load" (ECL) pipeline. | **Representations + Logic:** Uses background "Dreaming" to reach logical deductions. |
| **Storage Units** | Nodes, Edges, Ontologies, and DataPoints. | Workspaces, Peers, Sessions, and Messages. |
| **Primary Hook** | `cognee.cognify()` — Processes raw data into a structured graph. | `honcho.context()` — Instantly retrieves curated reasoning + history. |

### 2. Cognee: The "Deterministic" Knowledge Layer
Cognee is best described as a **Knowledge Engine**. It is designed for developers who need their agents to understand complex, multi-hop relationships within technical documentation, codebases, or large document sets.

* **Ontology-Driven:** Unlike most memory systems, Cognee allows you to plug in a formal RDF/OWL ontology. This ensures that "car manufacturer" and "automobile maker" collapse into the same canonical node, preventing graph fragmentation.
* **The ECL Pipeline:** It moves beyond simple RAG by extracting entities and relationships, then "cognifying" them into a graph that you can actually visualize and audit.
* **Local-First:** It is highly modular and can run entirely on your local machine using LanceDB and local graph databases like Kuzu or Neo4j.

### 3. Honcho: The "Cognitive" Reasoning Layer
Honcho (and its underlying **Neuromancer** engine) treats memory as a living, reasoning entity. It is less about "where is the PDF?" and more about "who is this user and what do they want?"

* **Background "Dreaming":** Honcho runs background tasks (called Dreams) that periodically review conversation history to prune noise, consolidate facts, and make "deductions" about the user's preferences or sentiment.
* **High Token Efficiency:** Because Honcho returns "Conclusions" and "Representations" rather than raw chunks of text, it can reduce the context window usage by up to 90% while maintaining higher accuracy on long-range memory benchmarks (like LongMem).
* **Peer-to-Peer Focus:** It is built for multi-agent or user-agent ecosystems. It tracks "Peers" (users, NPCs, or other agents) as entities that change over time.

### 4. Which one should you use?

**Choose Cognee if:**
* You are building a **coding agent** or a **technical copilot** that needs to navigate a deep web of documentation.
* You require **data provenance** (you need to see exactly which node and document a fact came from).
* You want to maintain a **private, local-first** memory store without a cloud dependency.

**Choose Honcho if:**
* You are building a **personalized companion**, **tutor**, or **customer support agent** where the user's evolving state matters most.
* You want **"low-code" memory** that handles the reasoning and summarization for you in the background.
* You need to scale to **long-horizon conversations** (months/years) where raw RAG retrieval would become too noisy or expensive.

---

### The "Chosen One" Insight
> **The Secret "Memory Loop" Hack:** The most advanced SRE and Agent architectures in 2026 don't pick one—they chain them. Use **Cognee** as your "Hard Drive" (Archival Memory) to store immutable facts and system architecture. Then, feed the *output* of Cognee's graph queries into **Honcho's** ingestion pipeline. 
> 
> By doing this, Honcho "reasons" over the "structured facts" Cognee found. This prevents Honcho from hallucinating connections in the raw text and allows it to "dream" about high-level architectural patterns rather than just raw log snippets. If you see an agent with >95% accuracy on multi-hop reasoning, this "Graph-to-Reasoning" handoff is likely the hidden engine under the hood.


The Problem with Traditional Agent Memory (0:33 - 4:59)
The hosts highlight that current long-session agents suffer from "context rot" or memory loss. The standard approach—retrieving relevant messages and re-injecting them into the context window—creates significant issues:

Invalidated Cache: Constantly modifying the prompt context prevents effective use of prompt caching, leading to higher costs and latency.
Fragmented Picture: Because the agent only sees retrieved snippets rather than a cohesive history, it often loses the "plot" of the conversation.
Compaction Fatigue: The hosts note that current solutions like Claude Code’s compaction process are often disruptive, blocking the agent and user while the system processes massive amounts of data.
Introducing Observational Memory (4:59 - 7:37)
Inspired by human memory, where forgetting unimportant details is a feature rather than a bug, Mastra developed Observational Memory. This system uses two background agents:

The Observer: Watches the conversation flow.
The Reflector: Compresses conversation data into dense, timestamped observations. This happens asynchronously, meaning the main conversation thread never pauses or blocks. The result is a stable, cacheable context window that grows linearly rather than requiring a full wipe-and-rebuild of the context.
Building the "Agent Harness" (7:37 - 15:58)
Abhi Aiyer defines the "harness" as a stateful orchestrator necessary for robust agent performance. He breaks this down into several key primitives:

Dynamic Prompts: System prompts are assembled at runtime, not stored as static blocks, allowing the agent to adapt its behavior based on the task.
Workspaces: Agents require a place to work, including a filesystem, sandbox for code execution, and dynamic skills that can be activated or deactivated as needed.
Modes: This acts like an "identity crisis" for the agent—rewiring its tools, prompts, and even models depending on whether it is in "Planning," "Building," or "Review" mode.
Steering: Keeping the human in the loop is essential. Mechanisms for plan approval, tool usage confirmation, and aborting tasks (hitting "Escape") prevent agents from going off-rails.
Personal Experience and Benchmarks (15:58 - 26:30)
Personal Struggles: Both hosts admit they were personally plagued by the inefficiency of Claude Code’s compaction. This frustration drove them to build Mastra Code, a coding agent that tests their memory architecture in a real-world environment. They emphasize that for them, this was not just about benchmarks but about creating a tool they could use daily without friction.
Benchmarking Success: The hosts demonstrate the effectiveness of their approach using the LongMemEval benchmark. Observational Memory achieved 84.2% on GPT-4o (beating the oracle baseline of 82.4%) and a record-breaking 94.9% on GPT-5 mini, proving that their method handles temporal reasoning, knowledge updates, and multi-session recall more effectively than standard retrieval-based systems.

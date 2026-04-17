# AI Agent Engineering: The Definitive Field Guide
>
> Synthesized from OpenCode, LangChain DeepAgents, and Anthropic's Agent Harness research.  
> For engineers who want to understand what's beneath the abstraction — not just make it work.

---

## Table of Contents

1. [The Mental Model: What an Agent Actually Is](#1-the-mental-model)
2. [Architecture Before Prompting](#2-architecture-before-prompting)
3. [Context Engineering: The Core Discipline](#3-context-engineering)
4. [Tool Design: The Primary Interface](#4-tool-design)
5. [The Agent Loop](#5-the-agent-loop)
6. [Multi-Agent Patterns](#6-multi-agent-patterns)
7. [The Long-Running Agent Harness](#7-long-running-agent-harness)
8. [Verification & Self-Correction](#8-verification--self-correction)
9. [Memory Architecture](#9-memory-architecture)
10. [Safety Primitives](#10-safety-primitives)
11. [Observability](#11-observability)
12. [Anti-Patterns Catalogue](#12-anti-patterns-catalogue)
13. [Strategic Reference Card](#13-strategic-reference-card)

---

## 1. The Mental Model

### What an Agent Actually Is

An AI agent is **not** a chatbot with tools. It is a control loop that uses an LLM as a reasoning engine to drive state transitions in the real world. The LLM is the CPU; context is RAM; tools are syscalls; the file system is disk.

```
┌─────────────────────────────────────────────────┐
│                  AGENT RUNTIME                  │
│                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │ GATHER   │───▶│  REASON  │───▶│  ACT     │   │
│  │ CONTEXT  │    │ (LLM)    │    │ (Tools)  │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│       ▲                                  │      │
│       └──────────── VERIFY ◀─────────────┘      │
│                                                 │
│  Persistence Layer: filesystem / SQLite / git   │
└─────────────────────────────────────────────────┘
```

### The Key Insight from Anthropic

> Give the agent a computer, not an API client.

The reason Claude Code works for non-coding tasks (research, video creation, note-taking) is that bash + filesystem + code generation is a **universal action space**. If your agent can write files, run scripts, and read output — it can do almost anything digital.

---

## 2. Architecture Before Prompting

### The Principle

A well-architected agent with mediocre prompts outperforms a poorly-architected agent with excellent prompts. Structure determines what's possible; prompts only tune what's probable.

### Separation of Concerns: Read vs. Write Modes

OpenCode's Build/Plan agent split is the canonical pattern. Never conflate analysis with execution at the architecture level.

```go
// Go — OpenCode-style agent mode definition
type AgentMode string

const (
    ModePlan  AgentMode = "plan"   // read-only, analysis, no side-effects
    ModeBuild AgentMode = "build"  // full tool access, can mutate state
)

type AgentConfig struct {
    Mode        AgentMode
    Tools       []Tool
    Permissions PermissionMap
    MaxIter     int
}

func NewAgent(mode AgentMode) *AgentConfig {
    switch mode {
    case ModePlan:
        return &AgentConfig{
            Mode:    ModePlan,
            Tools:   readOnlyTools(),    // ls, read, grep, search
            MaxIter: 20,
        }
    case ModeBuild:
        return &AgentConfig{
            Mode:    ModeBuild,
            Tools:   allTools(),         // + write, exec, bash, edit
            MaxIter: 50,
        }
    }
    return nil
}
```

### Decoupled Client/Server Architecture (OpenCode Pattern)

The UI layer must never entangle with the agent runtime. This enables headless operation, piping, scripting, and testing.

```
┌──────────────┐     JSON/RPC      ┌──────────────────────────┐
│   TUI / CLI  │ ◀───────────────▶ │    Agent Server          │
│  (Bubble Tea │                   │  - Session management    │
│   or any UI) │                   │  - Tool execution        │
└──────────────┘                   │  - LLM streaming         │
                                   │  - SQLite persistence    │
                                   └──────────────────────────┘
```

```python
# Python — Minimal agent server skeleton
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class AgentRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "build"  # "plan" | "build"

@app.post("/agent/run")
async def run_agent(req: AgentRequest):
    session = SessionStore.get_or_create(req.session_id)
    agent = AgentFactory.create(mode=req.mode, session=session)
    
    # Non-blocking — UI polls or streams
    task = asyncio.create_task(agent.run(req.message))
    return {"task_id": task.get_name(), "session_id": req.session_id}

@app.get("/agent/stream/{task_id}")
async def stream_output(task_id: str):
    # SSE stream back to client
    ...
```

---

## 3. Context Engineering

### The Fundamental Constraint

The context window is not working memory — it is a **bottleneck**. Every architectural decision in a well-built agent is, at its core, a context budget decision.

```
CONTEXT WINDOW BUDGET
┌──────────────────────────────────────────────────────┐
│ System Prompt / CLAUDE.md    ~10-15% of budget       │
│ Tool Definitions              ~15-20% (grows fast!)  │
│ Conversation History          ~30-40%                │
│ Tool Results / File Reads     ~20-30%                │
│ Working Space for Reasoning   ~10-15%  ← PROTECT THIS│
└──────────────────────────────────────────────────────┘
```

### Rules for Context Efficiency

**1. Agentic Search > Semantic Search (by default)**

Start with `grep`, `find`, `ls` — agentic filesystem search. Only add vector embedding / semantic search if you have proven latency requirements. Agentic search is transparent, debuggable, and requires zero infrastructure.

```python
# GOOD: Let the agent search agenically
tools = [
    BashTool(),       # grep -r "pattern" ./src
    ReadFileTool(),   # read specific file
    FindFileTool(),   # find . -name "*.py" -mtime -1
]

# BAD (premature): Add vector DB from day 1
tools = [
    VectorSearchTool(embedding_model="ada-002"),  # opaque, expensive, fragile
]
```

**2. Progressive Context Loading (Snippet → Full)**

Never load a full file if a snippet will answer the question. Structure your read tool to support range reading.

```go
// Go — Ranged file reader tool
type ReadFileTool struct{}

type ReadFileInput struct {
    Path      string `json:"path"`
    StartLine int    `json:"start_line,omitempty"` // 0 = beginning
    EndLine   int    `json:"end_line,omitempty"`   // 0 = full file
}

func (t *ReadFileTool) Execute(input ReadFileInput) (string, error) {
    lines, err := readLines(input.Path)
    if err != nil { return "", err }
    
    if input.StartLine == 0 && input.EndLine == 0 {
        // Return first 50 lines + summary of rest — NOT the whole file
        preview := lines[:min(50, len(lines))]
        return fmt.Sprintf("%s\n\n[%d more lines — use start_line/end_line to read more]",
            strings.Join(preview, "\n"),
            len(lines)-50,
        ), nil
    }
    return strings.Join(lines[input.StartLine:input.EndLine], "\n"), nil
}
```

**3. System Prompt Discipline**

Anthropic found Claude Agent SDK's system prompt consumes ~50 individual instructions before any user prompt, tools, or session content. Keep your CLAUDE.md/system prompt to universally applicable rules only. Task-specific instructions belong in separate, lazily-loaded markdown files.

```python
# BAD: Monolithic system prompt
SYSTEM_PROMPT = """
You are an agent. Here are all your rules:
[500 lines of mixed general + task-specific instructions]
"""

# GOOD: Minimal core + lazy-loaded skill files
SYSTEM_PROMPT = """
You are an autonomous agent with access to a computer.
Core rules:
- Work incrementally. One task at a time.
- Commit to git after each completed feature.
- Never mark a task complete without testing it.

Task-specific instructions are in ./skills/<task>.md — read the relevant one before starting.
"""
```

**4. Files as External Memory**

When context gets large, offload to the filesystem. The folder structure of your agent IS its extended memory architecture.

```
project/
├── AGENT_STATE.md         # Current task, last action, next steps
├── feature_list.json      # Structured task backlog (JSON not MD!)
├── claude-progress.txt    # Human-readable session log
├── init.sh               # How to start the environment
├── skills/               # Lazily-loaded instruction modules
│   ├── testing.md
│   ├── deployment.md
│   └── code-style.md
└── .git/                 # Recovery mechanism + history
```

---

## 4. Tool Design

### The Cardinal Rule

Tools are **prominent in the context window** — they are the first things the LLM considers when deciding what to do. Every tool you define is a standing invitation. Design accordingly.

### Tool Design Principles

```python
# GOOD TOOL: Self-contained, single responsibility, clear contract
class SearchEmailsTool:
    name = "search_emails"
    description = """
    Search emails by query string. Returns matching emails with subject, sender, date, and snippet.
    Use this as your PRIMARY way to find emails. For reading a specific email body, use read_email_by_id.
    """
    
    class Input(BaseModel):
        query: str = Field(description="Search terms, e.g. 'invoice from acme 2024'")
        max_results: int = Field(default=10, le=50, description="Limit results to avoid context bloat")
        date_after: Optional[str] = Field(default=None, description="ISO date string, e.g. '2024-01-01'")
    
    def execute(self, input: Input) -> list[dict]:
        # Returns structured, minimal data — not raw API response
        results = email_api.search(input.query, input.max_results)
        return [{"id": r.id, "subject": r.subject, "from": r.sender, 
                 "date": r.date, "snippet": r.snippet[:200]} for r in results]

# BAD TOOL: Overlapping responsibility, returns raw data, poor description
class EmailTool:
    name = "email"
    description = "Does email stuff"  # ← useless
    
    def execute(self, action: str, **kwargs):
        if action == "search": ...
        if action == "read": ...    # ← Multiple responsibilities = confusion
        if action == "send": ...
        return raw_api_response      # ← dumps everything into context
```

### Tool Anti-Patterns

```python
# ANTI-PATTERN 1: Tool that returns too much data
def get_all_logs() -> str:
    return open("app.log").read()  # Could be 50MB → instant context death

# FIX: Bounded, searchable
def search_logs(pattern: str, last_n_lines: int = 100) -> str:
    return subprocess.run(
        ["grep", "-n", pattern, "app.log", "|", "tail", f"-{last_n_lines}"],
        capture_output=True, text=True
    ).stdout

# ANTI-PATTERN 2: Tool with hidden side effects
def get_user_data(user_id: str) -> dict:
    data = db.get_user(user_id)
    audit_log.write(f"Accessed user {user_id}")  # ← Hidden side effect, agent doesn't know
    return data

# ANTI-PATTERN 3: Tool that requires a specific call sequence
def start_transaction(): ...
def update_record(): ...   # Must call start_transaction first — agent won't know
def commit(): ...
```

### Tool Permission Model (OpenCode Pattern)

```go
type Permission string

const (
    PermAsk   Permission = "ask"   // Always ask user before executing
    PermAllow Permission = "allow" // Execute without asking
    PermDeny  Permission = "deny"  // Never execute, return error
)

type PermissionMap map[string]Permission

var PlanModePermissions = PermissionMap{
    "read_file":    PermAllow,
    "search_files": PermAllow,
    "bash":         PermAsk,    // Even read-only bash needs approval in plan mode
    "write_file":   PermDeny,
    "delete_file":  PermDeny,
    "exec":         PermDeny,
}

var BuildModePermissions = PermissionMap{
    "read_file":    PermAllow,
    "search_files": PermAllow,
    "write_file":   PermAllow,
    "bash":         PermAllow,
    "delete_file":  PermAsk,   // Destructive ops always require approval
    "exec":         PermAllow,
}
```

---

## 5. The Agent Loop

### The Feedback Loop Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  AGENT LOOP                                                     │
│                                                                 │
│  1. ORIENT                                                      │
│     - Run pwd, read progress file, read git log                 │
│     - Load feature list, identify current state                 │
│     - Run init.sh / smoke test to verify environment is clean   │
│                                                                 │
│  2. PLAN (explicit, not implicit)                               │
│     - Update TODO list with next single task                    │
│     - Write plan to AGENT_STATE.md before acting               │
│                                                                 │
│  3. ACT                                                         │
│     - One feature at a time                                     │
│     - Use tools, write files, run code                         │
│                                                                 │
│  4. VERIFY                                                      │
│     - Lint / typecheck                                          │
│     - Run automated tests                                       │
│     - Visual/browser test if UI involved                        │
│     - Mark feature in feature_list.json ONLY if tests pass     │
│                                                                 │
│  5. PERSIST STATE                                               │
│     - git commit with descriptive message                       │
│     - Update claude-progress.txt                                │
│     - Update AGENT_STATE.md with next task                      │
│                                                                 │
│  6. CHECK LIMITS → if max_iter reached, stop cleanly           │
└─────────────────────────────────────────────────────────────────┘
```

### Implementing the Loop in Python

```python
import asyncio
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class LoopConfig:
    max_iterations: int = 50
    max_tokens_per_turn: int = 4096
    on_iteration: Callable[[int, Any], None] = None

class AgentLoop:
    def __init__(self, llm_client, tools: list, config: LoopConfig):
        self.llm = llm_client
        self.tools = {t.name: t for t in tools}
        self.config = config
        self.iteration = 0
        self.messages = []

    async def run(self, initial_prompt: str) -> str:
        self.messages = [{"role": "user", "content": initial_prompt}]
        
        while self.iteration < self.config.max_iterations:
            self.iteration += 1
            
            response = await self.llm.complete(
                messages=self.messages,
                tools=list(self.tools.values()),
                max_tokens=self.config.max_tokens_per_turn,
            )
            
            # If no tool calls — agent is done reasoning
            if not response.tool_calls:
                return response.content
            
            # Execute tool calls, collect results
            tool_results = []
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                if not tool:
                    result = f"ERROR: Unknown tool '{call.name}'"
                else:
                    try:
                        result = await tool.execute(**call.arguments)
                    except Exception as e:
                        result = f"ERROR: {type(e).__name__}: {e}"
                
                tool_results.append({
                    "tool_call_id": call.id,
                    "role": "tool",
                    "content": str(result),
                })
            
            # Append assistant message + tool results to history
            self.messages.append({"role": "assistant", **response.raw})
            self.messages.extend(tool_results)
            
            if self.config.on_iteration:
                self.config.on_iteration(self.iteration, response)
        
        # Max iterations reached — force terminal response
        self.messages.append({
            "role": "user",
            "content": "You have reached the maximum number of iterations. "
                       "Summarize what you've done and what remains."
        })
        final = await self.llm.complete(messages=self.messages)
        return final.content
```

### The Explicit Planning Tool (DeepAgents' TODO Trick)

Force the model to externalize its plan before acting. This is a structural intervention that dramatically improves long-horizon coherence.

```python
class TodoTool:
    """
    A no-op planning tool. Forces the model to commit to a plan in writing
    before taking any action. The list is written to disk so it persists
    across tool calls and is visible in logs.
    """
    name = "update_todo_list"
    description = """
    Update your task plan BEFORE starting work. Call this first in every session.
    Write the full list of subtasks you plan to complete, with status.
    This is NOT optional — use it to think step by step.
    """
    
    class Input(BaseModel):
        tasks: list[dict] = Field(
            description='[{"task": "description", "status": "pending|in_progress|done"}]'
        )
        current_focus: str = Field(description="The single task you are working on right now")
    
    def execute(self, input: Input) -> str:
        # Persist to disk — another agent session can pick this up
        with open("AGENT_STATE.md", "w") as f:
            f.write(f"# Current Focus\n{input.current_focus}\n\n")
            f.write("# Task List\n")
            for t in input.tasks:
                status_icon = {"pending": "[ ]", "in_progress": "[→]", "done": "[x]"}
                f.write(f"{status_icon.get(t['status'], '[ ]')} {t['task']}\n")
        
        return f"Plan saved. Current focus: {input.current_focus}"
```

---

## 6. Multi-Agent Patterns

### When to Use Sub-Agents

Sub-agents serve two distinct purposes — do not conflate them:

| Purpose | When to Use | What Sub-Agent Returns |
|---|---|---|
| **Parallelization** | Multiple independent queries that don't need each other's results | Final answers |
| **Context Isolation** | Deep investigation where intermediate steps would pollute main context | Cleaned, summarized output only |

### Supervisor-Researcher Pattern

```python
# The main agent (supervisor) stays clean
# Researchers dive deep in isolated contexts

class SupervisorAgent:
    """Orchestrates without getting polluted by detail."""
    
    async def run_research(self, topic: str) -> dict:
        # Spawn multiple isolated researcher agents in parallel
        sub_tasks = self.decompose_topic(topic)
        
        results = await asyncio.gather(*[
            ResearcherAgent().run(subtask, max_iter=20)
            for subtask in sub_tasks
        ])
        
        # Supervisor only sees CLEANED final outputs — not search history
        return self.synthesize(results)

class ResearcherAgent:
    """Goes deep on one subtopic. Burns its context freely."""
    
    async def run(self, subtopic: str, max_iter: int) -> str:
        # Has full web search, file read, analysis tools
        # At the end, returns a SUMMARY — not raw intermediate states
        raw_result = await agent_loop(subtopic, tools=research_tools, max_iter=max_iter)
        
        # CRITICAL: Compress before returning to supervisor
        return await compress_to_summary(raw_result, max_tokens=500)
```

### Context Isolation Rule

```python
# WRONG: Pass full context between agents
def orchestrate():
    result_a = agent_a.run(task_a)
    result_b = agent_b.run(task_b, context=result_a)  # ← passes EVERYTHING
    # Agent B now has all of Agent A's noise

# CORRECT: Pass only cleaned outputs
def orchestrate():
    result_a = agent_a.run(task_a)
    clean_a = extract_conclusions(result_a)  # strip intermediate reasoning
    result_b = agent_b.run(task_b, context=clean_a)  # ← only final facts
```

### Hidden System Agents (OpenCode Pattern)

Not all agents are user-facing. Some are architectural infrastructure:

```python
SYSTEM_AGENTS = {
    "compactor": {
        "trigger": "context_window > 0.80",  # fires at 80% context usage
        "prompt": "Summarize this conversation into the minimum context needed "
                  "to continue the task. Preserve all decisions, tool outputs, "
                  "and current state. Discard reasoning chains.",
        "user_visible": False,
    },
    "title_generator": {
        "trigger": "session_start AND message_count == 2",
        "prompt": "Generate a 5-word session title based on the task.",
        "user_visible": False,
    },
    "session_summarizer": {
        "trigger": "session_end",
        "prompt": "Write a summary of this session: what was done, "
                  "what state things are in, what comes next.",
        "output_file": "claude-progress.txt",
        "user_visible": False,
    }
}
```

---

## 7. The Long-Running Agent Harness

### The Problem: Amnesia Across Context Windows

Each new context window is a new agent with no memory of previous sessions. The harness is the scaffolding that gives every session a fast, reliable way to reconstruct state.

### Two-Agent Harness Design

```
Session 1 (Initializer Agent)
├── Reads user's high-level goal
├── Writes feature_list.json (ALL features marked "passes": false)
├── Writes init.sh (how to start the environment)
├── Writes claude-progress.txt (empty log)
├── Creates initial git commit
└── STOPS — does NOT start implementing

Session 2+ (Coding Agent)
├── Reads claude-progress.txt  ← where we left off
├── Reads git log --oneline -20  ← recent history
├── Reads feature_list.json  ← what's left to do
├── Runs init.sh  ← verify environment is healthy
├── Runs smoke test  ← catch any inherited bugs
├── Picks ONE feature from list
├── Implements + tests it
├── Marks feature "passes": true ONLY after real testing
├── git commit with descriptive message
├── Updates claude-progress.txt
└── STOPS — next session picks up here
```

### Feature List Structure (Use JSON, Not Markdown)

```json
{
  "features": [
    {
      "id": "F001",
      "category": "auth",
      "priority": 1,
      "description": "User can register with email and password",
      "acceptance_criteria": [
        "POST /auth/register returns 201 with user object",
        "Duplicate email returns 409",
        "Password is hashed, never stored plaintext",
        "Confirmation email is sent"
      ],
      "passes": false,
      "tested_at": null,
      "notes": ""
    }
  ]
}
```

> **Why JSON over Markdown?** LLMs are less likely to accidentally rewrite, restructure, or delete JSON. They treat it as data. Markdown is treated as editable prose.

### Initializer Agent Prompt (Template)

```python
INITIALIZER_PROMPT = """
You are setting up the environment for a long-running autonomous agent project.
Your job is NOT to implement anything — only to set up the scaffolding.

USER GOAL: {user_goal}

Complete these steps IN ORDER:

1. Create init.sh that:
   - Installs dependencies
   - Starts the development server
   - Runs a basic smoke test (start server, check health endpoint)

2. Create feature_list.json with:
   - EVERY feature the final product needs (be exhaustive)
   - All features marked "passes": false
   - Priority ordering (P1 = must-have, P2 = should-have, P3 = nice-to-have)
   - Acceptance criteria for each feature

3. Create claude-progress.txt with:
   - Project name and goal
   - Current status: "Initialized — ready for feature implementation"
   - Date initialized

4. Make an initial git commit:
   - Message: "chore: initialize project scaffolding"
   - Include all files you created

DO NOT implement any features. DO NOT write application code.
Stop after the git commit.
"""
```

### Coding Agent Prompt (Template)

```python
CODING_AGENT_PROMPT = """
You are a coding agent working on an existing project. A previous agent started this work.
Your job is to make incremental, clean progress and leave the environment better than you found it.

STARTUP SEQUENCE (run these in order before doing ANYTHING else):
1. Run `pwd` to confirm your working directory
2. Read claude-progress.txt to see what was done last session
3. Read git log --oneline -20 to see recent commits
4. Read feature_list.json to see what's pending
5. Run `bash init.sh` to start the environment
6. Run the smoke test to verify nothing is broken

WORK RULES:
- Work on exactly ONE feature per session
- Choose the highest-priority feature with "passes": false
- Only mark "passes": true after running real end-to-end tests
- Never remove or edit acceptance criteria — only change "passes" field
- Commit to git after completing each feature
- Commit message format: "feat: <feature description> (F<id>)"

END OF SESSION SEQUENCE:
1. Write a git commit with descriptive message
2. Update claude-progress.txt with:
   - What you completed this session
   - Current state of the environment
   - What the next agent should work on
3. Verify init.sh still works correctly

FAILURE RULES:
- If you find broken code from a previous session, fix it FIRST before new features
- Use `git revert` if you can't identify what broke things
- Never leave the environment in a broken state
"""
```

---

## 8. Verification & Self-Correction

### The Verification Hierarchy

```
Level 3: LLM-as-Judge (fuzzy, slow, expensive)
         ↑ use only when rules can't capture quality

Level 2: Visual / Browser Automation (for UI tasks)
         ↑ screenshots via Playwright/Puppeteer MCP

Level 1: Rules-Based (linting, type checking, unit tests)
         ↑ start here always

Level 0: Structural (does the file exist? does the server start?)
         ↑ the smoke test — must always pass
```

### Implementing Rules-Based Verification

```python
class VerificationTool:
    name = "verify_feature"
    description = "Run all verification checks for the feature just implemented. ALWAYS call this before marking a feature complete."
    
    class Input(BaseModel):
        feature_id: str
        feature_description: str
    
    async def execute(self, input: Input) -> dict:
        results = {}
        
        # Level 0: Smoke test
        results["smoke"] = await run_command("bash init.sh && curl -f http://localhost:3000/health")
        
        # Level 1a: Linting
        results["lint"] = await run_command("npm run lint 2>&1")
        
        # Level 1b: Type checking
        results["types"] = await run_command("npx tsc --noEmit 2>&1")
        
        # Level 1c: Unit tests related to this feature
        results["tests"] = await run_command(f"npm test -- --grep '{input.feature_id}' 2>&1")
        
        # Level 2: Browser automation (for UI features)
        if "ui" in input.feature_description.lower():
            results["visual"] = await run_playwright_test(input.feature_id)
        
        all_passed = all(r["exit_code"] == 0 for r in results.values())
        
        return {
            "passed": all_passed,
            "results": results,
            "verdict": "PASS — safe to mark feature complete" if all_passed 
                      else "FAIL — do not mark complete, fix issues first"
        }
```

### Visual Verification with Puppeteer MCP

```javascript
// Playwright MCP usage in agent context
// The agent calls these as tools, not as code it writes

async function verifyUIFeature(featureId, steps) {
    const browser = await playwright.chromium.launch();
    const page = await browser.newPage();
    
    await page.goto('http://localhost:3000');
    
    const screenshots = [];
    for (const step of steps) {
        await page.evaluate(step.action, step.params);
        const screenshot = await page.screenshot({ fullPage: false });
        screenshots.push({
            step: step.description,
            image: screenshot.toString('base64'),
        });
    }
    
    await browser.close();
    
    // Return screenshots to agent — it uses vision to verify
    return { feature_id: featureId, screenshots };
}
```

---

## 9. Memory Architecture

### Four Types of Agent Memory

```
┌──────────────────────────────────────────────────────────────┐
│ MEMORY TYPE    │ STORAGE       │ SCOPE          │ COST       │
├──────────────────────────────────────────────────────────────┤
│ In-Context     │ Messages array│ Current session│ Expensive  │
│ (Working)      │               │                │ (tokens)   │
├──────────────────────────────────────────────────────────────┤
│ External       │ Filesystem /  │ Persistent,    │ Free       │
│ (File System)  │ SQLite        │ cross-session  │            │
├──────────────────────────────────────────────────────────────┤
│ Semantic       │ Vector DB     │ Fuzzy recall   │ Medium     │
│ (Embeddings)   │               │ cross-session  │            │
├──────────────────────────────────────────────────────────────┤
│ Structured     │ DB / JSON     │ Precise recall │ Low        │
│ (State)        │               │ cross-session  │            │
└──────────────────────────────────────────────────────────────┘
```

### SQLite for Session Persistence (OpenCode Pattern)

```go
// Go — SQLite session store
type Session struct {
    ID        string    `db:"id"`
    Title     string    `db:"title"`
    CreatedAt time.Time `db:"created_at"`
    UpdatedAt time.Time `db:"updated_at"`
}

type Message struct {
    ID        int64     `db:"id"`
    SessionID string    `db:"session_id"`
    Role      string    `db:"role"`
    Content   string    `db:"content"`
    CreatedAt time.Time `db:"created_at"`
}

// Schema
const schema = `
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
`
```

### Automatic Context Compaction

```python
COMPACTION_THRESHOLD = 0.80  # Trigger at 80% context usage

class ContextManager:
    def __init__(self, max_tokens: int, model: str):
        self.max_tokens = max_tokens
        self.model = model
    
    async def maybe_compact(self, messages: list, current_usage: int) -> list:
        usage_ratio = current_usage / self.max_tokens
        
        if usage_ratio < COMPACTION_THRESHOLD:
            return messages  # No compaction needed
        
        # Keep system message + last N messages verbatim
        system_msgs = [m for m in messages if m["role"] == "system"]
        recent_msgs = messages[-6:]  # Keep last 3 turns intact
        old_msgs = messages[len(system_msgs):-6]
        
        # Compact the middle
        summary = await self.llm.complete(
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation history into the minimum "
                           f"context needed to continue the task. Preserve: "
                           f"1) All decisions made, 2) Current state, 3) What was just completed, "
                           f"4) What comes next. DISCARD: Reasoning chains, failed attempts.\n\n"
                           f"{json.dumps(old_msgs)}"
            }]
        )
        
        compacted = [{
            "role": "system",
            "content": f"[COMPACTED HISTORY]\n{summary.content}\n[END COMPACTED HISTORY]"
        }]
        
        return system_msgs + compacted + recent_msgs
```

---

## 10. Safety Primitives

### Hard Limits Are Not Optional

Every production agent needs these baked in, not bolted on:

```python
@dataclass
class AgentSafetyConfig:
    # Iteration limits
    max_iterations: int = 50           # Prevent runaway loops
    max_sequential_errors: int = 5     # Stop if tools keep failing
    
    # Cost limits
    max_tokens_total: int = 500_000    # Budget cap per session
    max_tokens_per_tool_result: int = 10_000  # Prevent context bombing
    
    # Time limits
    max_session_duration_seconds: int = 3600  # 1 hour max
    max_tool_execution_seconds: int = 30       # Per tool call timeout
    
    # Scope limits
    allowed_file_paths: list[str] = None   # If set, restrict file access
    denied_commands: list[str] = None       # e.g. ["rm -rf", "DROP TABLE"]
    require_approval_for: list[str] = None  # Tools requiring human in loop

class SafetyGuard:
    def __init__(self, config: AgentSafetyConfig):
        self.config = config
        self.iteration = 0
        self.sequential_errors = 0
        self.tokens_used = 0
        self.start_time = time.time()
    
    def check_before_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Returns (allowed, reason_if_denied)"""
        
        # Check iteration limit
        if self.iteration >= self.config.max_iterations:
            return False, f"Max iterations ({self.config.max_iterations}) reached"
        
        # Check time limit
        elapsed = time.time() - self.start_time
        if elapsed > self.config.max_session_duration_seconds:
            return False, "Session time limit exceeded"
        
        # Check token budget
        if self.tokens_used > self.config.max_tokens_total:
            return False, "Token budget exhausted"
        
        # Check denied commands
        if tool_name == "bash" and self.config.denied_commands:
            cmd = args.get("command", "")
            for denied in self.config.denied_commands:
                if denied in cmd:
                    return False, f"Command contains denied pattern: '{denied}'"
        
        # Check file path scope
        if tool_name in ("read_file", "write_file") and self.config.allowed_file_paths:
            path = args.get("path", "")
            if not any(path.startswith(allowed) for allowed in self.config.allowed_file_paths):
                return False, f"File path outside allowed scope: {path}"
        
        # Require human approval for sensitive operations
        if self.config.require_approval_for and tool_name in self.config.require_approval_for:
            approved = self.request_human_approval(tool_name, args)
            if not approved:
                return False, "User denied permission"
        
        return True, ""
```

---

## 11. Observability

### The Non-Negotiable: Trace Everything

```python
import structlog
from dataclasses import dataclass, field
from datetime import datetime

log = structlog.get_logger()

@dataclass
class AgentTrace:
    session_id: str
    iteration: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # LLM call
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Tool call (if any)
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result_tokens: int = 0
    tool_duration_ms: int = 0
    tool_error: str = ""
    
    # Agent state
    context_window_usage_pct: float = 0.0
    compaction_triggered: bool = False

class ObservableAgentLoop(AgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traces: list[AgentTrace] = []
        self.total_cost = 0.0
    
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        start = time.time()
        trace = AgentTrace(
            session_id=self.session_id,
            iteration=self.iteration,
            tool_name=tool_name,
            tool_args=args,
        )
        
        try:
            result = await self.tools[tool_name].execute(**args)
            trace.tool_duration_ms = int((time.time() - start) * 1000)
            
            log.info("tool_call",
                tool=tool_name,
                duration_ms=trace.tool_duration_ms,
                result_length=len(str(result)),
                session=self.session_id,
                iteration=self.iteration,
            )
            return result
            
        except Exception as e:
            trace.tool_error = str(e)
            log.error("tool_error", tool=tool_name, error=str(e), session=self.session_id)
            raise
        finally:
            self.traces.append(trace)
    
    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_iterations": self.iteration,
            "total_cost_usd": self.total_cost,
            "tool_calls": len(self.traces),
            "errors": sum(1 for t in self.traces if t.tool_error),
            "most_used_tools": Counter(t.tool_name for t in self.traces).most_common(5),
        }
```

### Key Metrics to Track

```python
AGENT_METRICS = {
    # Cost
    "tokens_per_task": "Total tokens consumed per completed task",
    "cost_per_iteration": "$ per LLM call — catches prompt bloat",
    
    # Reliability
    "task_completion_rate": "% of tasks completed without human intervention",
    "tool_error_rate": "% of tool calls that fail — indicates tool design issues",
    "compaction_frequency": "How often context overflows — indicates context design issues",
    
    # Quality
    "iterations_per_feature": "High = agent is confused; low = agent is efficient",
    "backtrack_rate": "How often agent reverts changes — indicates plan quality",
    
    # Safety
    "human_intervention_rate": "How often agent hits permission gates",
    "runaway_loop_count": "Max iteration limit hits — indicates prompt issues",
}
```

---

## 12. Anti-Patterns Catalogue

### The Fatal Seven

---

#### ❌ AP-1: One-Shotting Complex Tasks

```python
# WRONG
prompt = "Build me a full e-commerce platform with auth, payments, and inventory"
response = agent.run(prompt)  # Agent tries to do everything at once, fails halfway

# RIGHT
# Initializer: decompose into 40+ discrete features
# Each coding session: implement exactly one feature
```

**Root cause:** No feature decomposition. Agent fills its context implementing one giant thing, hits the limit mid-way, and the next session finds broken, half-implemented code.

---

#### ❌ AP-2: Victory Declaration Without Testing

```python
# WRONG — agent marks feature complete after writing code
feature["passes"] = True  # Code was written but never run end-to-end

# RIGHT — only mark complete after real test
result = await verify_feature_end_to_end(feature_id)
if result["all_checks_passed"]:
    feature["passes"] = True
    feature["tested_at"] = datetime.utcnow().isoformat()
```

**Root cause:** No mandatory verification gate. Agents optimize for "done" signals, not for actual correctness.

---

#### ❌ AP-3: Monolithic System Prompt

```python
# WRONG — 800 lines of mixed instructions
SYSTEM_PROMPT = open("everything.txt").read()

# RIGHT — minimal core, lazy-loaded skill modules
SYSTEM_PROMPT = """Core rules only (under 50 instructions)."""
# Load skill files into context only when relevant:
# "Read ./skills/testing.md before running any tests"
```

**Root cause:** Every instruction competes for attention in the context window. Too many instructions = agent follows none of them reliably.

---

#### ❌ AP-4: Passing Full Context Between Sub-Agents

```python
# WRONG
result_a = researcher_a.run(topic_a)
result_b = orchestrator.run(task, prior_context=result_a.full_trace)  # noise avalanche

# RIGHT  
summary_a = compress(result_a.full_trace, max_tokens=300)  # only conclusions
result_b = orchestrator.run(task, prior_context=summary_a)
```

**Root cause:** Intermediate reasoning of one agent is noise for another. Only conclusions have signal.

---

#### ❌ AP-5: No Iteration Limit

```python
# WRONG
while True:
    response = llm.complete(messages)
    if not response.tool_calls:
        break
    # ^ What if the agent loops forever calling the same tool?

# RIGHT
MAX_ITER = 50
for i in range(MAX_ITER):
    response = llm.complete(messages)
    if not response.tool_calls:
        break
else:
    # Force terminal: "You've hit the limit, summarize and stop."
    force_terminal_response(messages)
```

**Root cause:** LLMs can get stuck in tool-call loops (especially on errors). Without a hard limit, this burns through budget indefinitely.

---

#### ❌ AP-6: Semantic Search from Day One

```python
# WRONG (premature)
class AgentTools:
    def __init__(self):
        self.vector_db = Pinecone(api_key=..., index="my-agent")  # complex, opaque
        self.embedder = OpenAIEmbeddings(model="ada-002")         # another dependency
        # Now you have two additional failure surfaces before your agent even runs

# RIGHT (start here)
class AgentTools:
    def search_files(self, pattern: str, path: str = ".") -> str:
        return subprocess.run(
            ["grep", "-rn", pattern, path], capture_output=True, text=True
        ).stdout[:5000]  # bounded output
    # Add semantic search only when grep is provably too slow or imprecise
```

**Root cause:** Over-engineering the retrieval layer before proving the agent loop works.

---

#### ❌ AP-7: Tool Overlap / Ambiguity

```python
# WRONG — agent doesn't know which to use
tools = [
    SearchTool(),           # "search for information"
    WebSearchTool(),        # "search the web"
    DocumentSearchTool(),   # "search documents"
    KnowledgeBaseTool(),    # "search knowledge base"
]

# RIGHT — mutually exclusive, clearly scoped
tools = [
    LocalFileSearchTool(),  # searches ./project/** only
    WebSearchTool(),        # searches the open web
]
# Internally, LocalFileSearch uses grep; WebSearch uses Serper API
# Zero overlap in scope — agent always knows which to pick
```

**Root cause:** Overlapping tool descriptions force the LLM to guess, leading to inconsistent and unpredictable behavior.

---

## 13. Strategic Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              AI AGENT ENGINEERING CHEAT SHEET               │
├─────────────────────────────────────────────────────────────┤
│ DESIGN ORDER                                                │
│  Context budget → Tool contracts → Agent loop → Prompts    │
│                                                             │
│ CONTEXT RULES                                               │
│  Agentic search first. Semantic search only if proven.     │
│  Files = external memory. Filesystem IS the architecture.  │
│  System prompt < 50 instructions. Rest = lazy-loaded.      │
│  JSON > Markdown for agent-managed structured data.        │
│                                                             │
│ TOOL RULES                                                  │
│  One responsibility per tool. No overlapping scopes.        │
│  Bound all outputs. Never return raw API responses.         │
│  Destructive ops = PermAsk. Read ops = PermAllow.          │
│  Every tool must justify its context window cost.           │
│                                                             │
│ LOOP RULES                                                  │
│  Orient → Plan (explicit TODO) → Act → Verify → Persist   │
│  One feature per session. Commit after each.               │
│  Smoke test at session start. Always.                      │
│  Never mark done without end-to-end test.                  │
│                                                             │
│ MULTI-AGENT RULES                                           │
│  Sub-agents for parallelization OR context isolation.      │
│  Pass only conclusions upward. Never full intermediate.    │
│  Hidden system agents: compactor, titler, summarizer.      │
│  Separate plan agents from execution agents.               │
│                                                             │
│ SAFETY RULES                                                │
│  Hard iteration limit. Non-negotiable.                     │
│  Token budget per session. Prevents runaway cost.          │
│  Per-tool timeout. Prevents hung tool calls.               │
│  Allowed path scope for filesystem agents.                 │
│                                                             │
│ OBSERVABILITY RULES                                         │
│  Trace every LLM call: tokens, cost, duration.            │
│  Trace every tool call: name, args, result size, error.   │
│  Track compaction events. High frequency = design flaw.   │
│  Export to structured logs. Don't rely on console.        │
│                                                             │
│ ANTI-PATTERN TRIGGERS                                       │
│  Agent loops > 10x on same tool → prompt redesign          │
│  Compaction > 3x per task → context architecture issue    │
│  Tool error rate > 10% → tool design issue                │
│  "Task complete" without test run → verification missing   │
└─────────────────────────────────────────────────────────────┘
```

---

## Strategic Edge

**What practitioners who "use" agents don't understand but you should:**

1. **The filesystem is the agent's mind palace.** The folder structure you define is not a convenience — it is the agent's cognitive architecture. Every well-designed agentic system has a coherent filesystem schema that maps to the agent's task model. Design it the same way you'd design a database schema: before you write a line of agent code.

2. **JSON is a control surface, Markdown is prose.** Use JSON for any file the agent needs to read, update, and reason about structurally (feature lists, state files, task queues). Use Markdown for human-readable logs and documentation only. LLMs treat Markdown as free-form editable text and will rewrite it; they treat JSON as data and preserve its structure.

3. **The initializer/coding split is the agent equivalent of schema migrations.** Your initializer agent is your schema migration tool — it sets up the invariants everything else depends on. Your coding agents are application code — they operate within those invariants. Conflating the two is as dangerous as running schema migration code in your application runtime.

4. **Verification gates are your only reliability primitive.** Unlike software where tests run in milliseconds and can be run thousands of times, agent tasks are long, expensive, and non-idempotent. Your verification gate (the moment before marking something "done") is the single most important control point in the entire system. Make it expensive and thorough. This is where quality is produced or lost.

5. **Tool count is inversely correlated with agent reliability (up to a point).** Each tool added to the context increases the agent's option space and reduces the probability it will choose the right tool. Below ~15 tools, more tools = more capable. Above ~15, the agent starts making poor tool selection decisions. If you have 20+ tools, split into specialized sub-agents each with ≤10.

6. **The real differentiation in production agents is cross-session continuity design** — not prompting, not model choice. The teams shipping reliable autonomous systems have solved the "shift handoff problem": every session start takes < 3 LLM calls to fully reconstruct state. If your agent needs more than 5 tool calls just to understand where it is, your state persistence architecture is the bottleneck, not your model.

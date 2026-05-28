# Multi Agent Patterns

## Why Multi-Agent

65. Multi-agent systems coordinate specialized components to tackle complex workflows, but not every complex task requires this approach. A single agent with the right tools and prompt can sometimes achieve the same goal.

Reasons people usually want multi-agent systems:

- context management
- distributed development
- parallelization

## Main Patterns

26. Four architectural patterns:

- Subagents (Supervisor Pattern)
- Handoffs Pattern
- Skills Pattern
- Router Architecture

27 and 65. Additional patterns table:

Pattern | How it works
--- | ---
Subagents | Main agent coordinates subagents as tools
Handoffs | Tool calls update state and routing changes behavior dynamically
Skills | Specialized prompts or knowledge are loaded on demand
Router | Input is classified and directed to one or more specialized agents
Custom workflow | Bespoke LangGraph execution mixing deterministic and agentic behavior

## Communication Patterns

27. Agents usually exchange information through:

- Shared State
- Tool-Based Communication

19. Best practice for multi-agent systems is to use `AIMessage` objects with `tool_calls` to hand off tasks.

18. Circular delegation is possible. Loop safety is limited and may depend on the surrounding orchestration system.

## Pattern Trade-Offs

Subagents:

- strong for distributed development and parallelization
- weak for direct user-to-subagent interaction

Handoffs:

- strong for multihop conversations and direct interaction
- weaker for independent distributed development

Skills:

- strong for progressive disclosure and bounded context loading

Router:

- strong for parallelization
- weak for multihop sequences

Custom workflow:

- recommended when production control flow must be explicitly designed for the domain

# best practice for MCP tools

if you’re building a tool that you want other agents to use, you should consider shipping an MCP server.
it’s worth looking at building an MCP client that could access third-party features.
P0 Safeguards (Immediate): (12:45) Token-through (do not pass user tokens), check token expiry/audience, no public listeners (0.0.0.0), signed connectors only, and human-in-the-loop for destructive actions.
<https://youtu.be/bvuaF0B9vfA?si=x1KsfjpjbLxxTFpv>

1. **Focus on Intent, Not Operations** (0:43): Design MCP tools around the user's intent (e.g., "track order") rather than exposing individual operations (e.g., "get user by email," "get last order"). The MCP tool should handle the underlying complexity.
2. **Flatten Arguments**(2:05): Avoid using dictionaries for MCP tool arguments as this can lead to agent hallucination. Instead, declare specific, flattened arguments to make it easier for the agent to use.
3. **Instructions are Context** (4:15): The LLM (Large Language Model) uses not only tool names but also descriptions, argument hints, and even the tool's internal code to understand its purpose and how to use it effectively. Provide clear error messages and success information.
4. **Curate Ruthlessly** (5:04): Limit MCP servers to a maximum of 10 tools to prevent bloated context for the LLM. Each MCP server should have a single job, and unused or low-usage tools should be deleted. Consider splitting tools by persona (e.g., user vs. admin).
5.**Naming Tools** (5:54): Prefix tool names with the server name (e.g., "linear create issue" instead of "create issue") to avoid confusion when multiple servers might have similarly named functions.
5. **Implement Pagination**(6:41): Just like with APIs, MCP servers should support pagination for large results. Provide arguments for pagination (e.g., offset, limit) and return relevant information like total counts to the agent.

6. **Focus on Outcomes, Not Operations**: Stop forcing agents to orchestrate multiple tool calls; give them one high-level, outcome-oriented tool.

7. **Flatten Your Arguments**: Avoid nested structures and use constrained types like Literals to prevent hallucinations.

8. **Instructions are Context**: Treat your docstrings and error messages as direct instructions for the agent to self-correct.

9. **Curate Ruthlessly**: Keep servers focused with only 5–15 tools to save the agent’s context window.

10. **Name for Discovery**: Use service-prefixed names (e.g., slack_send_message) so agents can find the right tool quickly.

11. **Paginate Results**: Never dump large data sets; use metadata like has_more to keep the context clean.
12. **Sandboxed Env**: If it has filesystem access/Network access sandboxxing is a good idea.

## the challenge of context bloat when using the Model Context Protocol (MCP) at scale

As organizations build agents with access to hundreds of tools, presenting all these tools simultaneously to an agent's context window degrades performance and increases hallucinations. Their solution is Progressive Tool Discovery

The Problem: Context Bloat
**The Paradox of Choice**: Similar to being overwhelmed by too many streaming options, providing agents with hundreds of tools at once reduces their ability to select the correct ones for a specific task (1:27 - 1:50).
**Performance Degradation**: When an agent's context window is cluttered with irrelevant tool definitions, overall efficiency drops significantly (4:55 - 5:07).
**The Proposed Solution**: Progressive Tool Discovery
**Dynamic Loading**: Instead of exposing all tools upon initialization, the team built a mechanism where agents only load tools relevant to their specific problem category (e.g., 'operations' or 'monitoring') (6:55 - 7:05).
**Mechanism Implementation**:
**Initialization**: The agent starts with a single 'find tools' meta-tool (8:21 - 8:44).
**Categorization**on: Tools are mapped to specific problem spaces on the server side (8:28 - 8:40).
**Discovery**: When an agent needs to perform a task, it calls 'find tools' to request a specific category. The server persists the agent's context (session ID tracking) and returns the relevant tool subset (9:07 - 10:14).
**Updates**: Using the notifications/tools/list_changed feature of the MCP specification, the server pushes updates to the agent, which then fetches the new, leaner tool list (7:12 - 8:17, 9:46 - 10:02).
Key Takeaways
**Governance & Scaling**: While the technical solution helps, it still requires strong governance to ensure tool definitions are clear and categorized logically to avoid overlapping functionality (18:23 - 19:25).
**Flexibility**: The approach is compatible with both remote MCP servers (using server-side events/HTTP streaming) and local implementations using standard I/O (16:20 - 16:44).
**Future Improvements**: The team aims to move beyond their current deterministic (many-to-one) mapping toward more advanced, semantic searching where an agent can describe its problem in natural language to discover relevant tooling (16:01 - 16:15).

## Evolution of MCP Adoption

**From Reference to SaaS** (02:11-03:36): After launching in November 2024, the ecosystem moved beyond basic reference servers (SQL, Git, File system) to robust SaaS integrations like Slack, Clickhouse, and Notion.
**Creative Experiments** (03:36-04:21): Developers have creatively extended MCP to non-traditional domains, including Blender, Ableton, 3D printers, and even fantasy sports trackers.
**The Enterprise Backbone** (04:21-05:45): A critical, yet less visible, trend is the deployment of MCP behind corporate firewalls to bridge AI agents with sensitive internal systems like Jira, Salesforce, and Snowflake.
Protocol & Governance Milestones
**Technical Maturation**(05:45-07:28): The protocol has evolved to support remote servers, secure authorization, elicitations, and structured outputs for advanced agentic operations.
**Extensions & Apps**(07:28-08:14): The introduction of MCP Extensions allows for experimental features, while MCP Apps enables servers to render interactive UI patterns within client interfaces.
**Foundation Donation** (08:14-08:54): To ensure long-term stability and neutrality, MCP was donated to the Agentic AI Foundation.

The 2026 Roadmap: Productionizing AI Agents
**Transport Upgrades** (10:38-12:18): To handle hyperscale deployments, the protocol is shifting from streamable HTTP to a stateless HTTP model.
**Autonomous Tasks**(12:18-13:44): A new "tasks" primitive is being refined to standardize agentic communication for long-running, autonomous work.
**Enterprise Security**(13:44-14:56): Future updates include cross-app access, which enables seamless, authenticated connections to enterprise identity providers without requiring manual OAuth flows.

Future Horizons & Ecosystem Growth
**Upcoming Features**(14:56-16:35): The horizon includes triggers (webhooks for proactively notifying clients), native streaming for incremental tool results, and skills to bundle domain-specific knowledge.
**SDK v2**(16:35-17:44): Rewritten Python and TypeScript SDKs will launch soon to improve developer ergonomics and remove legacy technical debt.
**Solving Context Bloat** (17:44-19:46): Parra addresses criticism regarding context consumption by advocating for progressive discovery and tool search, ensuring agents only load tools when strictly necessary.
**Composability**(19:46-21:07): The team is encouraging "composability through code," where clients use interpreters to execute code that chains MCP tools together, leveraging structured outputs to reason about complex return types.

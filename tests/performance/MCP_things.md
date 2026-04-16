# best practice for MCP tools

if you’re building a tool that you want other agents to use, you should consider shipping an MCP server.
it’s worth looking at building an MCP client that could access third-party features.
P0 Safeguards (Immediate): (12:45) Token-through (do not pass user tokens), check token expiry/audience, no public listeners (0.0.0.0), signed connectors only, and human-in-the-loop for destructive actions.
<https://youtu.be/bvuaF0B9vfA?si=x1KsfjpjbLxxTFpv>

1. Focus on Intent, Not Operations (0:43): Design MCP tools around the user's intent (e.g., "track order") rather than exposing individual operations (e.g., "get user by email," "get last order"). The MCP tool should handle the underlying complexity.
2. Flatten Arguments (2:05): Avoid using dictionaries for MCP tool arguments as this can lead to agent hallucination. Instead, declare specific, flattened arguments to make it easier for the agent to use.
3. Instructions are Context (4:15): The LLM (Large Language Model) uses not only tool names but also descriptions, argument hints, and even the tool's internal code to understand its purpose and how to use it effectively. Provide clear error messages and success information.
4. Curate Ruthlessly (5:04): Limit MCP servers to a maximum of 10 tools to prevent bloated context for the LLM. Each MCP server should have a single job, and unused or low-usage tools should be deleted. Consider splitting tools by persona (e.g., user vs. admin).
5. Naming Tools (5:54): Prefix tool names with the server name (e.g., "linear create issue" instead of "create issue") to avoid confusion when multiple servers might have similarly named functions.
6. Implement Pagination (6:41): Just like with APIs, MCP servers should support pagination for large results. Provide arguments for pagination (e.g., offset, limit) and return relevant information like total counts to the agent.

7. Focus on Outcomes, Not Operations: Stop forcing agents to orchestrate multiple tool calls; give them one high-level, outcome-oriented tool.

8. Flatten Your Arguments: Avoid nested structures and use constrained types like Literals to prevent hallucinations.

9. Instructions are Context: Treat your docstrings and error messages as direct instructions for the agent to self-correct.

10. Curate Ruthlessly: Keep servers focused with only 5–15 tools to save the agent’s context window.

11. Name for Discovery: Use service-prefixed names (e.g., slack_send_message) so agents can find the right tool quickly.

12. Paginate Results: Never dump large data sets; use metadata like has_more to keep the context clean.

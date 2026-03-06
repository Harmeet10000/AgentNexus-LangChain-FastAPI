from mcp.server.fastmcp import FastMCP
from my_server.tools.github import search_repos

# Name your server clearly; this appears in LLM logs
mcp = FastMCP("InsightEngine")

# Register tools by importing them
mcp.add_tool(search_repos)

if __name__ == "__main__":
    mcp.run()
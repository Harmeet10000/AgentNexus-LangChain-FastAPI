1. add Field description for tool instead of simple docstrings
2. use structured ouput everywhere for llm output, tool output, MCP output
3. use toons for serialisation before sending to LLM
4. use toons for deserialisation after receiving from LLM
5. use toons for serialisation before sending to tools
6. use toons for deserialisation after receiving from tools
7. use async functions, methods and packages in langchain and langGraph
8. trim/remove tool output in a multi step agent conversation 
9. # CORRECT — use dedicated package imports
from langchain_tavily import TavilySearch 
# WRONG — deprecated community import path
from langchain_community.tools.tavily_search import TavilySearchResults
10. all methods, functions, model and agent invocation should have langsmith decorator for proper obervability
11. always normalise agent state after fetching from checkpointer so that there is no version mismatch
12. have proper retry mechanism for tools with idenpotent execution as mention in langchain docs
13.  

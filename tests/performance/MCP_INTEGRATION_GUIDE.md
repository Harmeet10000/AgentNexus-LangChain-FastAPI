# MCP (Model Context Protocol) Integration Guide for LangChain

## Overview
This guide demonstrates how to integrate MCP (Model Context Protocol) servers with LangChain agents in your FastAPI application, enabling dynamic tool discovery and multi-server communication.

---

## 📦 Installation

```bash
# Install MCP adapters for LangChain
pip install langchain-mcp-adapters

# Or using uv
uv add langchain-mcp-adapters
```

---

## 🏗️ Project Structure

```
src/
├── mcp/                              # MCP Integration (NEW)
│   ├── __init__.py
│   ├── client.py                    # MCP client configuration
│   ├── servers/                     # MCP server implementations
│   │   ├── __init__.py
│   │   ├── math_server.py           # Math operations server
│   │   ├── weather_server.py        # Weather data server
│   │   ├── database_server.py       # Database query server
│   │   └── filesystem_server.py     # File system server
│   ├── adapters/                    # Tool adapters
│   │   ├── __init__.py
│   │   └── tool_adapter.py
│   └── config/
│       ├── __init__.py
│       └── server_config.py         # Server configurations
```

---

## 1. 🔧 MCP Client Setup

### 1.1 Basic MCP Client Configuration

```python
# src/mcp/client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Dict, Any, List
from langchain.tools import BaseTool
import asyncio
import os

class MCPClientManager:
    """Manage MCP client and server connections."""
    
    def __init__(self, server_config: Dict[str, Dict[str, Any]]):
        """
        Initialize MCP client with server configurations.
        
        Args:
            server_config: Dictionary of server configurations
                {
                    "server_name": {
                        "transport": "stdio" | "streamable_http",
                        "command": "python",  # For stdio
                        "args": ["/path/to/server.py"],  # For stdio
                        "url": "http://localhost:5000/mcp"  # For http
                    }
                }
        """
        self.server_config = server_config
        self.client: MultiServerMCPClient = None
        self._tools: List[BaseTool] = None
    
    async def initialize(self):
        """Initialize the MCP client and connect to servers."""
        self.client = MultiServerMCPClient(self.server_config)
        await self.client.start()
        self._tools = await self.client.get_tools()
        return self
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools from connected MCP servers."""
        if self._tools is None:
            self._tools = await self.client.get_tools()
        return self._tools
    
    async def get_tools_by_server(self, server_name: str) -> List[BaseTool]:
        """Get tools from a specific server."""
        all_tools = await self.get_tools()
        return [tool for tool in all_tools if tool.metadata.get("server") == server_name]
    
    async def cleanup(self):
        """Clean up MCP client connections."""
        if self.client:
            await self.client.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.cleanup())
```

### 1.2 Server Configuration

```python
# src/mcp/config/server_config.py
import os
from pathlib import Path

# Get absolute paths
BASE_DIR = Path(__file__).parent.parent
SERVERS_DIR = BASE_DIR / "servers"

# MCP Server Configurations
MCP_SERVER_CONFIG = {
    # Local subprocess-based servers
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": [str(SERVERS_DIR / "math_server.py")],
        "enabled": True,
    },
    
    "filesystem": {
        "transport": "stdio",
        "command": "python",
        "args": [str(SERVERS_DIR / "filesystem_server.py")],
        "enabled": True,
    },
    
    "database": {
        "transport": "stdio",
        "command": "python",
        "args": [str(SERVERS_DIR / "database_server.py")],
        "enabled": os.getenv("ENABLE_DB_MCP", "false").lower() == "true",
    },
    
    # HTTP-based remote servers
    "weather": {
        "transport": "streamable_http",
        "url": os.getenv("WEATHER_MCP_URL", "http://localhost:5000/mcp"),
        "enabled": os.getenv("ENABLE_WEATHER_MCP", "false").lower() == "true",
    },
    
    "external_api": {
        "transport": "streamable_http",
        "url": os.getenv("EXTERNAL_MCP_URL", "http://api.example.com/mcp"),
        "enabled": os.getenv("ENABLE_EXTERNAL_MCP", "false").lower() == "true",
    },
}

def get_enabled_servers() -> Dict[str, Dict[str, Any]]:
    """Get only enabled server configurations."""
    return {
        name: {k: v for k, v in config.items() if k != "enabled"}
        for name, config in MCP_SERVER_CONFIG.items()
        if config.get("enabled", False)
    }
```

---

## 2. 🖥️ MCP Server Implementations

### 2.1 Math Server Example

```python
# src/mcp/servers/math_server.py
#!/usr/bin/env python3
"""MCP server for mathematical operations."""

import asyncio
import json
import sys
from typing import Any, Dict
import math

# Simple MCP server implementation
class MathMCPServer:
    """MCP server providing math operations."""
    
    def __init__(self):
        self.tools = {
            "add": self.add,
            "subtract": self.subtract,
            "multiply": self.multiply,
            "divide": self.divide,
            "power": self.power,
            "sqrt": self.sqrt,
        }
    
    async def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    async def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    async def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    async def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    async def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent
    
    async def sqrt(self, x: float) -> float:
        """Calculate square root of x."""
        if x < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(x)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        tool_name = request.get("tool")
        params = request.get("params", {})
        
        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            result = await self.tools[tool_name](**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    async def run(self):
        """Run the MCP server."""
        # Read from stdin, write to stdout
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                
                # Write response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    server = MathMCPServer()
    asyncio.run(server.run())
```

### 2.2 Weather Server Example (HTTP-based)

```python
# src/mcp/servers/weather_server.py
"""HTTP-based MCP server for weather information."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import httpx

app = FastAPI()

class MCPRequest(BaseModel):
    """MCP request schema."""
    tool: str
    params: Dict[str, Any]

class MCPResponse(BaseModel):
    """MCP response schema."""
    result: Optional[Any] = None
    error: Optional[str] = None

class WeatherService:
    """Weather data service."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "demo_key"
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    async def get_current_weather(self, city: str, country: str = None) -> Dict:
        """Get current weather for a city."""
        location = f"{city},{country}" if country else city
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/weather",
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch weather data"
                )
            
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
            }
    
    async def get_forecast(self, city: str, days: int = 5) -> Dict:
        """Get weather forecast for a city."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/forecast",
                params={
                    "q": city,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": days * 8  # 8 forecasts per day
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch forecast data"
                )
            
            return response.json()

weather_service = WeatherService()

@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest) -> MCPResponse:
    """Handle MCP requests."""
    try:
        if request.tool == "get_current_weather":
            result = await weather_service.get_current_weather(**request.params)
            return MCPResponse(result=result)
        
        elif request.tool == "get_forecast":
            result = await weather_service.get_forecast(**request.params)
            return MCPResponse(result=result)
        
        else:
            return MCPResponse(
                error=f"Unknown tool: {request.tool}",
                result={"available_tools": ["get_current_weather", "get_forecast"]}
            )
    
    except Exception as e:
        return MCPResponse(error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "weather_mcp"}
```

### 2.3 Database Server Example

```python
# src/mcp/servers/database_server.py
#!/usr/bin/env python3
"""MCP server for database operations."""

import asyncio
import json
import sys
from typing import Any, Dict, List
from motor.motor_asyncio import AsyncIOMotorClient
import os

class DatabaseMCPServer:
    """MCP server for database queries."""
    
    def __init__(self):
        mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.client = AsyncIOMotorClient(mongo_url)
        self.db = self.client[os.getenv("MONGODB_DATABASE", "langchain_db")]
    
    async def query_documents(
        self,
        collection: str,
        filter: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Query documents from a collection."""
        coll = self.db[collection]
        cursor = coll.find(filter or {}).limit(limit)
        documents = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return documents
    
    async def insert_document(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        """Insert a document into a collection."""
        coll = self.db[collection]
        result = await coll.insert_one(document)
        return str(result.inserted_id)
    
    async def update_document(
        self,
        collection: str,
        filter: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """Update documents in a collection."""
        coll = self.db[collection]
        result = await coll.update_many(filter, {"$set": update})
        return result.modified_count
    
    async def delete_document(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> int:
        """Delete documents from a collection."""
        coll = self.db[collection]
        result = await coll.delete_many(filter)
        return result.deleted_count
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        tool_name = request.get("tool")
        params = request.get("params", {})
        
        tools = {
            "query_documents": self.query_documents,
            "insert_document": self.insert_document,
            "update_document": self.update_document,
            "delete_document": self.delete_document,
        }
        
        if tool_name not in tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(tools.keys())
            }
        
        try:
            result = await tools[tool_name](**params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
    
    async def run(self):
        """Run the MCP server."""
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                
                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
                
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    server = DatabaseMCPServer()
    asyncio.run(server.run())
```

---

## 3. 🤖 Agent Integration with MCP

### 3.1 MCP-Enabled Agent

```python
# src/agents/mcp/mcp_agent.py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List
from src.mcp.config.server_config import get_enabled_servers

class MCPAgent:
    """Agent with MCP tool integration."""
    
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        self.mcp_client: MultiServerMCPClient = None
        self.agent = None
        self.memory = MemorySaver()
    
    async def initialize(self):
        """Initialize MCP client and agent."""
        # Get enabled server configurations
        server_config = get_enabled_servers()
        
        # Initialize MCP client
        self.mcp_client = MultiServerMCPClient(server_config)
        
        # Get tools from all MCP servers
        tools = await self.mcp_client.get_tools()
        
        # Initialize language model
        model = init_chat_model(self.model_name, model_provider="google")
        
        # Create agent with MCP tools
        self.agent = create_react_agent(
            model,
            tools,
            checkpointer=self.memory
        )
        
        return self
    
    async def run(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Execute agent with MCP tools."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config
        )
        
        return {
            "response": result["messages"][-1].content,
            "messages": result["messages"],
            "thread_id": thread_id
        }
    
    async def stream(self, query: str, thread_id: str = "default"):
        """Stream agent responses."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        async for chunk in self.agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="values"
        ):
            yield chunk
    
    async def cleanup(self):
        """Clean up MCP client."""
        if self.mcp_client:
            await self.mcp_client.stop()
```

### 3.2 Multi-Server Agent Example

```python
# examples/mcp_agent_example.py
import asyncio
from src.agents.mcp.mcp_agent import MCPAgent

async def main():
    """Example using MCP agent with multiple servers."""
    
    # Initialize agent
    agent = MCPAgent(model_name="gemini-pro")
    await agent.initialize()
    
    try:
        # Math query (uses math MCP server)
        print("Testing Math Server:")
        math_result = await agent.run(
            query="What's (3 + 5) x 12?",
            thread_id="math_session"
        )
        print(f"Math Result: {math_result['response']}\n")
        
        # Weather query (uses weather MCP server)
        print("Testing Weather Server:")
        weather_result = await agent.run(
            query="What is the weather in NYC?",
            thread_id="weather_session"
        )
        print(f"Weather Result: {weather_result['response']}\n")
        
        # Database query (uses database MCP server)
        print("Testing Database Server:")
        db_result = await agent.run(
            query="Query the users collection and show me the first 5 users",
            thread_id="db_session"
        )
        print(f"Database Result: {db_result['response']}\n")
        
        # Multi-tool query
        print("Testing Multi-Tool Query:")
        multi_result = await agent.run(
            query="Calculate 15 * 8, then check the weather in London",
            thread_id="multi_session"
        )
        print(f"Multi-Tool Result: {multi_result['response']}\n")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. 🌐 FastAPI Integration

### 4.1 MCP Agent Endpoint

```python
# src/api/endpoints/mcp_agents.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.agents.mcp.mcp_agent import MCPAgent
from fastapi.responses import StreamingResponse
import json

router = APIRouter(prefix="/mcp-agents", tags=["MCP Agents"])

# Global agent instance
mcp_agent: Optional[MCPAgent] = None

class MCPAgentRequest(BaseModel):
    """MCP agent request model."""
    query: str = Field(..., description="User query")
    thread_id: str = Field(default="default", description="Conversation thread ID")
    stream: bool = Field(default=False, description="Stream response")

class MCPAgentResponse(BaseModel):
    """MCP agent response model."""
    response: str
    thread_id: str
    metadata: Dict[str, Any] = {}

async def get_mcp_agent() -> MCPAgent:
    """Get or initialize MCP agent."""
    global mcp_agent
    
    if mcp_agent is None:
        mcp_agent = MCPAgent()
        await mcp_agent.initialize()
    
    return mcp_agent

@router.post("/execute", response_model=MCPAgentResponse)
async def execute_mcp_agent(
    request: MCPAgentRequest,
    agent: MCPAgent = Depends(get_mcp_agent)
):
    """Execute MCP agent with query."""
    try:
        result = await agent.run(
            query=request.query,
            thread_id=request.thread_id
        )
        
        return MCPAgentResponse(
            response=result["response"],
            thread_id=result["thread_id"],
            metadata={
                "message_count": len(result["messages"])
            }
        )
    
    except Exception as e:
        raise HTTPException(500, f"Agent execution failed: {str(e)}")

@router.post("/stream")
async def stream_mcp_agent(
    request: MCPAgentRequest,
    agent: MCPAgent = Depends(get_mcp_agent)
):
    """Stream MCP agent responses."""
    
    async def generate():
        try:
            async for chunk in agent.stream(
                query=request.query,
                thread_id=request.thread_id
            ):
                # Extract last message from chunk
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    data = {
                        "content": last_message.content if hasattr(last_message, "content") else str(last_message),
                        "type": last_message.type if hasattr(last_message, "type") else "unknown"
                    }
                    yield f"data: {json.dumps(data)}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@router.get("/servers")
async def list_mcp_servers():
    """List connected MCP servers."""
    from src.mcp.config.server_config import get_enabled_servers
    
    servers = get_enabled_servers()
    
    return {
        "servers": [
            {
                "name": name,
                "transport": config["transport"],
                "status": "connected"
            }
            for name, config in servers.items()
        ],
        "total": len(servers)
    }

@router.get("/tools")
async def list_mcp_tools(agent: MCPAgent = Depends(get_mcp_agent)):
    """List all available MCP tools."""
    tools = await agent.mcp_client.get_tools()
    
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "server": tool.metadata.get("server", "unknown")
            }
            for tool in tools
        ],
        "total": len(tools)
    }
```

### 4.2 Update Main Router

```python
# src/api/router.py
from fastapi import APIRouter
from src.api.endpoints import chat, rag, documents, crawl, workflows, mcp_agents

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(chat.router)
api_router.include_router(rag.router)
api_router.include_router(documents.router)
api_router.include_router(crawl.router)
api_router.include_router(workflows.router)
api_router.include_router(mcp_agents.router)  # NEW: MCP agents
```

---

## 5. 🔧 Environment Configuration

```bash
# .env additions for MCP
# Enable/disable MCP servers
ENABLE_DB_MCP=true
ENABLE_WEATHER_MCP=true
ENABLE_EXTERNAL_MCP=false

# MCP server URLs
WEATHER_MCP_URL=http://localhost:5000/mcp
EXTERNAL_MCP_URL=http://api.example.com/mcp

# Weather API key (if using weather server)
OPENWEATHER_API_KEY=your_api_key_here
```

---

## 6. 🧪 Testing MCP Integration

```python
# tests/test_mcp_integration.py
import pytest
from src.agents.mcp.mcp_agent import MCPAgent

@pytest.mark.asyncio
async def test_mcp_agent_math():
    """Test MCP agent with math server."""
    agent = MCPAgent()
    await agent.initialize()
    
    try:
        result = await agent.run("What is 5 + 3?")
        assert result["response"] is not None
        assert "8" in result["response"].lower()
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_mcp_agent_multi_tool():
    """Test MCP agent with multiple tools."""
    agent = MCPAgent()
    await agent.initialize()
    
    try:
        result = await agent.run(
            "Calculate 10 * 5 and tell me the weather in London"
        )
        assert result["response"] is not None
    finally:
        await agent.cleanup()
```

---

## 7. 📝 Usage Examples

### Example 1: Basic MCP Agent

```python
from src.agents.mcp.mcp_agent import MCPAgent

async def example():
    agent = MCPAgent()
    await agent.initialize()
    
    result = await agent.run("What's 25 * 4?")
    print(result["response"])
    
    await agent.cleanup()
```

### Example 2: Streaming Responses

```python
async def streaming_example():
    agent = MCPAgent()
    await agent.initialize()
    
    async for chunk in agent.stream("Calculate 100 / 5"):
        if "messages" in chunk:
            print(chunk["messages"][-1].content)
    
    await agent.cleanup()
```

### Example 3: HTTP Request

```bash
curl -X POST "http://localhost:5000/api/v1/mcp-agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is (15 + 25) * 3?",
    "thread_id": "session_123"
  }'
```

---

## 8. 🚀 Deployment Considerations

### Docker Compose

```yaml
# docker-compose.yml additions
services:
  app:
    # ... existing config ...
    environment:
      - ENABLE_DB_MCP=true
      - ENABLE_WEATHER_MCP=true
  
  weather-mcp:
    build:
      context: .
      dockerfile: docker/mcp/weather.Dockerfile
    ports:
      - "8001:5000"
    environment:
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
```

---

## 📚 Key Takeaways

1. **MultiServerMCPClient** - Manages connections to multiple MCP servers
2. **Transport Types** - Use `stdio` for local, `streamable_http` for remote
3. **Tool Discovery** - Automatically discover tools from all servers
4. **Agent Integration** - Seamlessly integrate with LangChain agents
5. **FastAPI Endpoints** - Expose MCP functionality via REST API

---

**Version**: 1.0  
**Last Updated**: 2025-10-04  
**Compatible with**: langchain-mcp-adapters latest

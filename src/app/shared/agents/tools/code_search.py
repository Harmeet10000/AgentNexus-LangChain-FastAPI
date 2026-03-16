import asyncio
from typing import cast

import httpx
from langchain_core.tools import BaseTool, tool
from loguru import logger
from redis.asyncio import Redis

from app.shared.tools.search_dtos import RipgrepSearchRequest, ShadowZoektSearchRequest
from app.utils.exceptions import ValidationException


def get_code_search_tools(redis_client: Redis, zoekt_url: str) -> list[BaseTool]:
    """
    Factory to inject dependencies into LangChain tools.
    Called by the Service layer where dependencies are resolved.
    """

    @tool(args_schema=RipgrepSearchRequest)
    async def ripgrep_search(
        regex_pattern: str, 
        file_extension: str | None = None, 
        max_results: int = 10
    ) -> str:
        """
        Executes a high-speed ripgrep search across the local filesystem.
        Use this for finding exact strings, error codes, or hardcoded values.
        """
        logger.bind(pattern=regex_pattern, ext=file_extension).info("Executing ripgrep tool")
        
        # Security: Prevent arbitrary command execution
        if pattern_contains_dangerous_flags(regex_pattern):
             raise ValidationException("Unsafe regex pattern detected.")

        cmd = ["rg", "--line-number", "--heading", "--max-count", str(max_results)]
        if file_extension:
            cmd.extend(["--type", file_extension])
        cmd.append(regex_pattern)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0 and process.returncode != 1:
            error_msg = stderr.decode().strip()
            logger.bind(error=error_msg).warning("Ripgrep execution failed")
            return f"Search failed: {error_msg}"

        result = stdout.decode().strip()
        return result if result else "No results found."

    @tool(args_schema=ShadowZoektSearchRequest)
    async def shadow_indexed_zoekt_search(
        symbol_name: str, 
        repo_name: str | None = None
    ) -> str:
        """
        Searches for code symbols (functions, classes) using the Shadow Indexing Protocol.
        It ranks results by recent modifications to surface highly relevant bugs/features.
        """
        logger.bind(symbol=symbol_name).info("Initiating Shadow Zoekt Search")

        # 1. The Shadow Indexing Protocol: Bloom Filter Check
        # Prevents hallucinations by checking if the symbol exists in the GST
        bloom_key = f"gst:bloom:{repo_name or 'global'}"
        
        try:
            # BF.EXISTS requires RedisBloom module
            exists = await redis_client.execute_command("BF.EXISTS", bloom_key, symbol_name)
            if not exists:
                logger.bind(symbol=symbol_name).debug("Bloom filter rejected symbol hallucination")
                return f"Symbol '{symbol_name}' does not exist in the Global Symbol Table. Do not attempt to search for it again."
        except Exception as e:
            # Graceful degradation if RedisBloom is unavailable
            logger.bind(error=str(e)).warning("Bloom filter check failed, bypassing.")

        # 2. Zoekt Search with Custom mtime Ranking
        # We explicitly request Zoekt to order by filesystem modified time
        query_parts = [symbol_name]
        if repo_name:
            query_parts.append(f"repo:{repo_name}")
            
        zoekt_query = " ".join(query_parts)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{zoekt_url}/api/search",
                    json={
                        "query": zoekt_query,
                        "order_by": "mtime",  # The 48-hour recency advantage
                        "max_results": 5
                    },
                    timeout=5.0
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("Result"):
                    return "No matching symbols found in index."
                    
                # Format output cleanly for the LLM
                formatted_results = []
                for match in data["Result"]["Files"]:
                    formatted_results.append(f"File: {match['FileName']}\nCode:\n{match['Lines']}")
                    
                return "\n---\n".join(formatted_results)
                
            except httpx.HTTPError as e:
                logger.bind(error=str(e)).error("Zoekt backend unreachable")
                return "Search backend is currently down."

    return [ripgrep_search, shadow_indexed_zoekt_search]

def pattern_contains_dangerous_flags(pattern: str) -> bool:
    """Helper to prevent command injection via malicious regex strings."""
    return pattern.startswith("-") or ";" in pattern or "|" in pattern
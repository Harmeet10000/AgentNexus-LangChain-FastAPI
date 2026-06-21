"""Video 11: Build a knowledge graph with LLM extraction.

Demonstrates:
- nested Pydantic schemas for entities and relationships
- extracting a small knowledge graph from a Wikipedia article
- saving the result to JSON

Prerequisites:
- `pip install crawl4ai playwright pydantic`
- `playwright install`
- `OPENAI_API_KEY` set in your environment

Run:
- `python crawl4ai_101/video_11_knowledge_graph.py`
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List

from crawl4ai import CacheMode, AsyncWebCrawler, CrawlerRunConfig, LLMConfig, LLMExtractionStrategy

API_KEY = os.getenv("OPENAI_API_KEY")
URL = "https://en.wikipedia.org/wiki/Web_scraping"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "video_11"

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


if BaseModel:
    class Entity(BaseModel):
        name: str
        description: str


    class Relationship(BaseModel):
        entity1: str
        entity2: str
        relation_type: str
        description: str


    class KnowledgeGraph(BaseModel):
        entities: List[Entity]
        relationships: List[Relationship]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


async def main() -> None:
    if not API_KEY:
        print("SKIP: set OPENAI_API_KEY to run the knowledge graph demo.")
        return
    if BaseModel is None:
        print("SKIP: install pydantic to run the knowledge graph demo.")
        return

    output_dir = ensure_output_dir()
    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token=API_KEY),
        schema=KnowledgeGraph.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "Extract the main entities and their relationships from this article. "
            "Keep the graph compact and factual."
        ),
        input_format="markdown",
        chunk_token_threshold=1200,
        extra_args={"temperature": 0, "max_tokens": 1200},
    )
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=strategy,
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)

    if not result.success:
        print(f"Knowledge graph crawl failed: {result.error_message}")
        return

    graph = json.loads(result.extracted_content or "{}")
    if isinstance(graph, list):
        graph = graph[0] if graph and isinstance(graph[0], dict) else {"entities": graph, "relationships": []}
    if not isinstance(graph, dict):
        graph = {"entities": [], "relationships": []}
    graph_path = output_dir / "knowledge_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])
    print(f"Knowledge graph saved to: {graph_path}")
    print(f"Entities: {len(entities)}, relationships: {len(relationships)}")
    if relationships:
        print(f"Sample relationship: {relationships[0]}")


if __name__ == "__main__":
    asyncio.run(main())

"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################


load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                "content": "Claude is the flagship AI assistant from Anthropic. It was previously "
                "known as Claude Instant in its earlier versions.",
                "type": EpisodeType.text,
                "description": "AI podcast transcript",
            },
            {
                "content": "As an AI assistant, Claude has been available since December 15, 2022 – Present",
                "type": EpisodeType.text,
                "description": "AI podcast transcript",
            },
            {
                "content": {
                    "name": "GPT-4",
                    "creator": "OpenAI",
                    "capability": "Multimodal Reasoning",
                    "previous_version": "GPT-3.5",
                    "training_data_cutoff": "April 2023",
                },
                "type": EpisodeType.json,
                "description": "AI model metadata",
            },
            {
                "content": {
                    "name": "GPT-4",
                    "release_date": "March 14, 2023",
                    "context_window": "128,000 tokens",
                    "status": "Active",
                },
                "type": EpisodeType.json,
                "description": "AI model metadata",
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f"AI Agents Unleashed {i}",
                episode_body=episode["content"]
                if isinstance(episode["content"], str)
                else json.dumps(episode["content"]),
                source=episode["type"],
                source_description=episode["description"],
                reference_time=datetime.now(UTC),
            )
            print(f'Added episode: AI Agents Unleashed {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Which AI assistant is from Anthropic?'")
        results = await graphiti.search("Which AI assistant is from Anthropic?")

        # Print search results
        print("\nSearch Results:")
        for result in results:
            print(f"UUID: {result.uuid}")
            print(f"Fact: {result.fact}")
            if hasattr(result, "valid_at") and result.valid_at:
                print(f"Valid from: {result.valid_at}")
            if hasattr(result, "invalid_at") and result.invalid_at:
                print(f"Valid until: {result.invalid_at}")
            print("---")

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print("\nReranking search results based on graph distance:")
            print(f"Using center node UUID: {center_node_uuid}")

            reranked_results = await graphiti.search(
                "Which AI assistant is from Anthropic?",
                center_node_uuid=center_node_uuid,
            )

            # Print reranked search results
            print("\nReranked Search Results:")
            for result in reranked_results:
                print(f"UUID: {result.uuid}")
                print(f"Fact: {result.fact}")
                if hasattr(result, "valid_at") and result.valid_at:
                    print(f"Valid from: {result.valid_at}")
                if hasattr(result, "invalid_at") and result.invalid_at:
                    print(f"Valid until: {result.invalid_at}")
                print("---")
        else:
            print("No results found in the initial search to use as center node.")

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            "\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:"
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query="Large Language Models",
            config=node_search_config,
        )

        # Print node search results
        print("\nNode Search Results:")
        for node in node_search_results.nodes:
            print(f"Node UUID: {node.uuid}")
            print(f"Node Name: {node.name}")
            node_summary = (
                node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
            )
            print(f"Content Summary: {node_summary}")
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f"Created At: {node.created_at}")
            if hasattr(node, "attributes") and node.attributes:
                print("Attributes:")
                for key, value in node.attributes.items():
                    print(f"  {key}: {value}")
            print("---")

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(main())

from graphiti_core.driver.neo4j_driver import Neo4jDriver

# Create a Neo4j driver with custom database name
driver = Neo4jDriver(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    database="my_custom_database",  # Custom database name
)

# Pass the driver to Graphiti
graphiti = Graphiti(graph_driver=driver)

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

# Google API key configuration
api_key = "<your-google-api-key>"

# Initialize Graphiti with Gemini clients
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "password",
    llm_client=GeminiClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.0-flash")
    ),
    embedder=GeminiEmbedder(
        config=GeminiEmbedderConfig(api_key=api_key, embedding_model="embedding-001")
    ),
    cross_encoder=GeminiRerankerClient(
        config=LLMConfig(api_key=api_key, model="gemini-2.5-flash-lite")
    ),
)

# Now you can use Graphiti with Google Gemini for all components

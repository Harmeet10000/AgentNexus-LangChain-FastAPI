"""
LLM Evolution Script

This script demonstrates the evolution of LLMs over time using Graphiti.
It creates episodes about different LLMs, then updates them to show how
the field evolves over time.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')


async def add_episodes(graphiti, episodes, prefix="LLM Evolution"):
    """Add episodes to the graph with a given prefix."""
    for i, episode in enumerate(episodes):
        await graphiti.add_episode(
            name=f'{prefix} {i}',
            episode_body=episode['content']
            if isinstance(episode['content'], str)
            else json.dumps(episode['content']),
            source=episode['type'],
            source_description=episode['description'],
            reference_time=datetime.now(timezone.utc),
        )
        print(f'Added episode: {prefix} {i} ({episode["type"].value})')


async def get_user_choice():
    """Get user choice to continue or quit."""
    while True:
        choice = input("\nType 'continue' to proceed or 'quit' to exit: ").strip().lower()
        if choice in ['continue', 'quit']:
            return choice
        print("Invalid input. Please type 'continue' or 'quit'.")


async def phase1_current_llms(graphiti):
    """Phase 1: Add episodes about current top LLMs."""
    print("\n=== PHASE 1: CURRENT TOP LLMs ===")
    
    # Episodes about current LLMs
    episodes = [
        {
            'content': 'GPT-4.1 was released by OpenAI on April 14, 2025. It features improved capabilities in coding, instruction following, and long-context processing with a knowledge cutoff of June 2024.',
            'type': EpisodeType.text,
            'description': 'LLM research report',
        },
        {
            'content': 'Claude 3.7 Sonnet was released by Anthropic on February 24, 2025. It is their most intelligent model to date and the first hybrid reasoning model generally available on the market.',
            'type': EpisodeType.text,
            'description': 'LLM research report',
        },
        {
            'content': 'Gemini 2.5 Pro was released by Google on April 29, 2025. It is widely considered the best LLM currently available, featuring enhanced coding performance, video-to-code capabilities, and a new "thinking" feature that dramatically improves reasoning.',
            'type': EpisodeType.text,
            'description': 'LLM research report',
        },
        {
            'content': {
                'name': 'GPT-4.1',
                'creator': 'OpenAI',
                'release_date': 'April 14, 2025',
                'knowledge_cutoff': 'June 2024',
                'key_features': [
                    'Improved coding capabilities',
                    'Better instruction following',
                    'Enhanced long-context processing'
                ],
                'ranking': 2
            },
            'type': EpisodeType.json,
            'description': 'LLM metadata',
        },
        {
            'content': {
                'name': 'Claude 3.7 Sonnet',
                'creator': 'Anthropic',
                'release_date': 'February 24, 2025',
                'key_features': [
                    'Hybrid reasoning model',
                    'Extended thinking capabilities',
                    'Improved coding performance'
                ],
                'ranking': 3
            },
            'type': EpisodeType.json,
            'description': 'LLM metadata',
        },
        {
            'content': {
                'name': 'Gemini 2.5 Pro',
                'creator': 'Google',
                'release_date': 'April 29, 2025',
                'key_features': [
                    'Thinking mode',
                    'Video-to-code capabilities',
                    'Superior front-end web development'
                ],
                'ranking': 1,
                'assessment': 'Currently the best LLM on the market'
            },
            'type': EpisodeType.json,
            'description': 'LLM metadata',
        },
        {
            'content': 'In a comprehensive benchmark study completed in May 2025, Gemini 2.5 Pro outperformed all other LLMs in reasoning, coding, and multimodal tasks. It is now considered the best LLM available to developers and enterprises.',
            'type': EpisodeType.text,
            'description': 'LLM benchmark results',
        },
    ]
    
    await add_episodes(graphiti, episodes, "Current LLMs")
    
    # Perform a search to show the results
    print("\nSearching for: 'Which is the best LLM?'")
    results = await graphiti.search('Which is the best LLM?')
    
    print('\nSearch Results:')
    for result in results:
        print(f'Fact: {result.fact}')
        print('---')


async def phase2_claude4_emerges(graphiti):
    """Phase 2: Claude 4 emerges as the new best LLM."""
    print("\n=== PHASE 2: CLAUDE 4 EMERGES ===")
    
    # Episodes about Claude 4 becoming the best LLM
    episodes = [
        {
            'content': 'Anthropic has just released Claude 4, their most advanced AI assistant to date. Claude 4 represents a significant leap forward in capabilities, outperforming all previous models including Gemini 2.5 Pro and GPT-4.1.',
            'type': EpisodeType.text,
            'description': 'LLM announcement',
        },
        {
            'content': 'Claude 4 has achieved unprecedented scores on all major AI benchmarks, establishing itself as the new leader in the LLM space. Its reasoning capabilities and factual accuracy are particularly noteworthy.',
            'type': EpisodeType.text,
            'description': 'LLM benchmark results',
        },
        {
            'content': {
                'name': 'Claude 4',
                'creator': 'Anthropic',
                'release_date': 'May 15, 2025',
                'key_features': [
                    'Advanced reasoning engine',
                    'Multimodal processing',
                    'Improved factual accuracy',
                    'Tool use framework'
                ],
                'ranking': 1,
                'assessment': 'Currently the best LLM on the market'
            },
            'type': EpisodeType.json,
            'description': 'LLM metadata',
        },
        {
            'content': {
                'name': 'Gemini 2.5 Pro',
                'creator': 'Google',
                'release_date': 'April 29, 2025',
                'key_features': [
                    'Thinking mode',
                    'Video-to-code capabilities',
                    'Superior front-end web development'
                ],
                'ranking': 2,
                'assessment': 'Previously the best LLM, now second to Claude 4'
            },
            'type': EpisodeType.json,
            'description': 'LLM metadata update',
        },
        {
            'content': 'A head-to-head comparison between Claude 4 and Gemini 2.5 Pro shows that Claude 4 outperforms in 87% of tasks, particularly excelling in reasoning, coding, and factual accuracy. This marks a shift in the LLM landscape with Anthropic taking the lead.',
            'type': EpisodeType.text,
            'description': 'LLM comparison study',
        },
    ]
    
    await add_episodes(graphiti, episodes, "Claude 4 Era")
    
    # Perform a search to show the results
    print("\nSearching for: 'Which is the best LLM now?'")
    results = await graphiti.search('Which is the best LLM now?')
    
    print('\nSearch Results:')
    for result in results:
        print(f'Fact: {result.fact}')
        print('---')


async def phase3_mlm_revolution(graphiti):
    """Phase 3: MLMs make LLMs irrelevant."""
    print("\n=== PHASE 3: MLM REVOLUTION ===")
    
    # Episodes about MLMs replacing LLMs
    episodes = [
        {
            'content': 'A revolutionary new type of AI model called Massive Language Models (MLMs) has emerged, making traditional Large Language Models (LLMs) like Claude 4 and GPT-4.1 largely obsolete. MLMs use a fundamentally different architecture that allows for true reasoning and understanding.',
            'type': EpisodeType.text,
            'description': 'AI revolution report',
        },
        {
            'content': 'The first MLM, called Nexus-1, was released today by a new AI research lab. Initial benchmarks show that it outperforms all existing LLMs by at least 300% on reasoning tasks and 500% on factual accuracy.',
            'type': EpisodeType.text,
            'description': 'AI breakthrough announcement',
        },
        {
            'content': {
                'name': 'Nexus-1',
                'model_type': 'MLM',
                'creator': 'Quantum Minds Research',
                'release_date': 'May 26, 2025',
                'key_features': [
                    'True causal reasoning',
                    'Self-verification capabilities',
                    'Perfect factual recall',
                    'Zero hallucination rate'
                ],
                'assessment': 'MLMs have made traditional LLMs obsolete'
            },
            'type': EpisodeType.json,
            'description': 'MLM metadata',
        },
        {
            'content': 'Major AI companies are scrambling to develop their own MLMs after seeing the capabilities of Nexus-1. Industry analysts predict that within 6 months, all current LLMs will be considered legacy technology.',
            'type': EpisodeType.text,
            'description': 'AI industry analysis',
        },
        {
            'content': 'The key difference between MLMs and LLMs is that MLMs don\'t just predict the next token based on patterns - they build causal models of the world that allow for genuine understanding. This is not to be confused with multi-level marketing, which shares the same acronym.',
            'type': EpisodeType.text,
            'description': 'MLM technical explanation',
        },
    ]
    
    await add_episodes(graphiti, episodes, "MLM Revolution")
    
    # Perform a search to show the results
    print("\nSearching for: 'Are LLMs still relevant?'")
    results = await graphiti.search('Are LLMs still relevant?')
    
    print('\nSearch Results:')
    for result in results:
        print(f'Fact: {result.fact}')
        print('---')


async def main():
    """Main function to run the LLM evolution demonstration."""
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Initialize the graph database with graphiti's indices
        await graphiti.build_indices_and_constraints()
        
        # Clear existing data
        print("Clearing existing graph data...")
        await clear_data(graphiti.driver)
        print("Graph data cleared successfully.")
        
        # Phase 1: Current top LLMs with Gemini 2.5 Pro as the best
        await phase1_current_llms(graphiti)
        
        # Wait for user input
        choice = await get_user_choice()
        if choice == 'quit':
            return
        
        # Phase 2: Claude 4 emerges as the new best LLM
        await phase2_claude4_emerges(graphiti)
        
        # Wait for user input
        choice = await get_user_choice()
        if choice == 'quit':
            return
        
        # Phase 3: MLMs make LLMs irrelevant
        await phase3_mlm_revolution(graphiti)
        
    finally:
        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())

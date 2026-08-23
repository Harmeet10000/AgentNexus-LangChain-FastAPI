# Cognee v1 API

## Purpose
Long-term episodic and procedural memory for Agent Saul via the cognee v1 API: remember/improve/recall replace the deprecated add/cognify/search surface.

## Requirements

### Requirement: Store content via remember
The system SHALL use `cognee.remember()` to store content in episodic memory, replacing the deprecated `cognee.add()`.

#### Scenario: Store final report
- **WHEN** `store_final_report()` is called with `report_json` and `dataset_name`
- **THEN** `cognee.remember(report_json, dataset_name=dataset_name)` is called

#### Scenario: Store relationships
- **WHEN** `store_relationships()` is called with `relationships_text` and `dataset_name`
- **THEN** `cognee.remember(relationships_text, dataset_name=dataset_name)` is called

### Requirement: Process content via improve
The system SHALL use `cognee.improve()` to process and enrich stored content, replacing the deprecated `cognee.cognify()`.

#### Scenario: Process report after store
- **WHEN** `store_final_report()` stores content successfully
- **THEN** `cognee.improve(dataset=dataset_name)` is called

#### Scenario: Process relationships after store
- **WHEN** `store_relationships()` stores content successfully
- **THEN** `cognee.improve(dataset=dataset_name)` is called

### Requirement: Query memory via recall
The system SHALL use `cognee.recall()` to query stored memories, replacing the deprecated `cognee.search()`. The query type SHALL be auto-routed by Cognee (default `auto_route=True`) — no explicit `SearchType` enum is required.

#### Scenario: Search episodic memory
- **WHEN** `search_episodic_memory()` is called with a `query` and `user_id`
- **THEN** `cognee.recall(query_text=query, datasets=[dataset_name])` is called

#### Scenario: Search returns results as dicts
- **WHEN** `cognee.recall()` returns a list of results
- **THEN** each result is converted to a dict and returned as a list

#### Scenario: Search handles failures gracefully
- **WHEN** `cognee.recall()` raises an exception
- **THEN** an empty list is returned and the error is logged

### Requirement: No type ignore suppressions
The system SHALL NOT use `# type: ignore` comments on cognee API calls. The Cognee 1.0 API provides proper type stubs.

#### Scenario: Type checker passes
- **WHEN** `uv run ty check src/app/shared/langchain_layer/agents/memory/cognee_client.py` is run
- **THEN** no type errors are reported on `cognee.remember()`, `cognee.improve()`, or `cognee.recall()` calls

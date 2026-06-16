# Capability: llm-injection

## Purpose

Inject LLM client via constructor instead of creating per-request.

## ADDED Requirements

### Requirement: SearchService Constructor Injection

The `SearchService.__init__` SHALL take `llm: BaseChatModel` as required parameter and `ask_legal()` SHALL use `self._llm` instead of creating new `ChatGoogleGenerativeAI`. The `ChatGoogleGenerativeAI` import SHALL be removed from `search/service.py`.

#### Scenario: SearchService receives LLM via constructor

- Given a `BaseChatModel` instance
- When `SearchService` is constructed with the LLM
- Then `ask_legal()` uses the injected LLM
- And no `ChatGoogleGenerativeAI` import exists in search/service.py

### Requirement: Document Service Injection

Document processing functions SHALL take `llm: BaseChatModel` parameter. `build_document_ingestion_graph` SHALL accept `llm` param and thread it through the closure to `process_document_ingestion`.

#### Scenario: Document functions receive LLM via parameter

- Given a `BaseChatModel` instance
- When `process_document_ingestion` is called with the LLM
- Then the injected LLM is used for generation
- And `build_document_ingestion_graph` passes LLM through to inner functions

### Requirement: Dependency Layer Updates

`search/dependencies.py` SHALL create LLM via `_build_chat_model()` once and pass to `SearchService`. `documents/dependencies.py` SHALL create LLM once and pass to `DocumentQueryService`.

#### Scenario: Dependency layer creates LLM once and injects

- Given `get_search_service` is called
- When the dependency is resolved
- Then `_build_chat_model()` is called once
- And the same LLM instance is passed to `SearchService`

### Requirement: Backward Compatibility

No breaking changes SHALL be made to API request/response contracts. This is internal refactoring only.

#### Scenario: API contracts unchanged

- Given an existing API endpoint
- When the endpoint is called
- Then the response shape is identical to before the refactor

## Non-Goals

- Async LLM factory
- LLM caching/memoization
- Model selection logic changes

# Production-Grade Python/FastAPI/LangChain Template - Coding Standards & Examples

## Project Overview

This is a production-grade AI-powered service built with Python, FastAPI, LangChain, and modern async patterns following a **features-based layered architecture** with enterprise-level patterns focusing on security, scalability, and maintainability.
- Don't leave comments in code, unless they explain something complex and not trivial

## Architecture Principles

- **Features-Based Organization**: Each feature is self-contained with complete layer stack
- **Async-First**: Use async/await patterns throughout for better performance
- **Layered Architecture**: Router → Service → Repository → Model
- **Separation of Concerns**: Clear boundaries between layers
- **DRY Principle**: Don't repeat yourself, create reusable utilities
- **Type Safety**: Use Pydantic models and type hints everywhere

## Code Style Examples

### Rep

### S

### Router Layer (FastAPI)

### Correlation ID Middleware

### Updated Router Examples with Standardized Responses

### Pydantic Models/Schemas


### Database Models (SQLAlchemy)


### LangChain Integration Patterns


### Logging Standards


## Coding Standards

### General Rules

- Use Python 3.12+ features and type hints everywhere
- Follow PEP 8 style guide with ruff formatter
- Use async/await for all I/O operations with asyncio/asyncer
- Prefer composition over inheritance
- Use Pydantic models for data structures
- Add docstrings for all public functions and classes

### Naming Conventions

- **Files**: `snake_case.py`
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Environment Variables**: `UPPER_SNAKE_CASE`

### Error Handling

### Global Exception Handler


### Dep

## Feature Structure Template

```
src/features/[feature-name]/
├── __init__.py
├── router.py              # FastAPI route definitions
├── service.py             # Business logic and LangChain integration
├── repository.py          # Data access layer
├── schemas.py             # Pydantic models
├── models.py              # SQLAlchemy models
├── dependencies.py        # FastAPI dependencies
├── exceptions.py          # Feature-specific exceptions
└── constants.py           # Feature constants
```

### Configuration Management



## Development Workflow

1. Create feature directory under `src/app/features/`
2. Implement layered architecture within feature
3. Add comprehensive Pydantic model validation
4. Include OpenAPI documentation via FastAPI
5. Add feature routes to main app routing
6. Write unit and integration tests with pytest
7. Run `ruff` and `ty` before commit
8. Update API documentation automatically via FastAPI

## Performance Considerations

- Use async/await throughout the application
- Implement proper database connection pooling
- Use Redis for caching and session management
- Optimize LangChain memory usage for conversations
- Monitor token usage and costs for AI operations
- Use database indexes effectively
- Implement proper pagination for large datasets

## Security Checklist

- [ ] Input validation with Pydantic models
- [ ] Authorization checks in place
- [ ] No hardcoded secrets (use environment variables)
- [ ] Proper error handling without information leakage
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers applied via middleware
- [ ] SQL injection prevention (use SQLAlchemy ORM)
- [ ] API key management for external services

## LangChain Best Practices

- Use conversation memory for chat ty
- Implement proper tounting and limits
- Handle API rate limits gracefully
- Use callbacks for monitoring and logging
- Implement proper error handling for AI operations
- Cache frequently used prompts and chains
- Monitor costs and usage metrics
- Use streaming for long responses when possible

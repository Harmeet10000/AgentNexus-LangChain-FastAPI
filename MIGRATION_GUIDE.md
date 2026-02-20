# Alembic Migration Commands

## Setup Complete ✅

### Database Structure Created:
- `src/database/base.py` - SQLAlchemy Base class
- `src/database/schemas/document_vectors.py` - DocumentVector model
- `src/database/schemas/chat_messages.py` - ChatMessage & ChatSession models
- `src/alembic/env.py` - Updated to import correct models

### Configuration Verified:
- ✅ Database URL loaded from `.env.development` (POSTGRES_URL)
- ✅ No hardcoded URLs in `alembic.ini`
- ✅ Settings properly imported in `postgres.py`

---



---

## Common Operations

### Create New Migration (after model changes)
```bash
alembic revision --autogenerate -m "Add new column to chat_messages"
```

### Rollback Last Migration
```bash
alembic downgrade -1
```

### Rollback to Specific Version
```bash
alembic downgrade <revision_id>
```

### Rollback All Migrations
```bash
alembic downgrade base
```

---

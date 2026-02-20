#!/bin/bash
alembic init <directory>
# Initializes Alembic in your project (creates alembic.ini + migration folder).
# Variants:
alembic init --template pyproject alembic # → modern setup using pyproject.toml
alembic init --template generic alembic # → classic alembic.ini style

alembic list_templates
# Shows available initialization templates (useful when deciding on setup style).

## Migration Commands

### 1. Check Configuration
```bash
# Verify Alembic can see your models
alembic check
```

### 2. Create Initial Migration
```bash
# Auto-generate migration from models
alembic revision --autogenerate -m "Initial schema: document_vectors and chat tables"
```

### 3. Review Generated Migration
```bash
# Check the generated file in src/alembic/versions/
ls -la src/alembic/versions/
```

### 4. Apply Migration
```bash
# Run the migration
alembic upgrade head
```

### 5. Check Current Version
```bash
# Show current migration version
alembic current
```

### 6. View Migration History
```bash
# Show all migrations
alembic history
```
# Migration Creation Commands

alembic revision --autogenerate -m "short description"Most used command: Detects model changes and generates a migration script automatically.
alembic revision -m "manual migration"
# Creates an empty migration file (for custom SQL, data changes, raw operations).

# Common useful flags:

--autogenerate or -a → compare models vs current DB
--sql → output raw SQL instead of Python ops (great for review or DBA hand-off)
--head → explicitly base on current head (rarely needed)
--depends-on <revision> → force dependency on specific revision

# Applying & Rolling Back Migrations

alembic upgrade headMost important command: Apply all pending migrations → bring DB to latest state.
alembic upgrade <revision>
# Apply up to a specific revision (e.g. alembic upgrade abc1234).
alembic downgrade -1
# Roll back the last migration step.
alembic downgrade base
# Roll back all migrations → dangerous in production!

# Other useful variants:

alembic upgrade head --sql > apply.sql → generate SQL script only
alembic downgrade -1 --sql > rollback.sql
alembic upgrade <rev1>:<rev2> → migrate from one revision to another

# Inspection & Debugging Commands

alembic current
# Show the current revision(s) applied in the database.
alembic history
# Show full list of migration revisions + messages (very useful!).
alembic heads
# Show current head revision(s) — important when branches exist.
alembic branches
# List branched revisions (happens when two people create migrations independently).
alembic show <revision>
# Display details of a single migration file (useful before applying).
alembic stamp head
# Mark current DB state as a specific revision without actually running ops (rescue tool).
alembic ensure_version
# Create the alembic_version table if missing (rarely needed manually).

# Branch & Conflict Management

alembic merge <rev1> <rev2> -m "merge two branches"
# Combine parallel migration branches into one new revision.

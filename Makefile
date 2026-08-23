.PHONY: help lint format type-check precommit test migrate-create migrate-up migrate-down migrate-current migrate-history celery celery-ingestion celery-beat celery-command

help:
	@echo "Available commands:"
	@echo "  make lint           - Run ruff linter with auto-fix"
	@echo "  make format         - Format code with ruff"
	@echo "  make type-check     - Run ty type checker"
	@echo "  make precommit      - Run pre-commit hooks on all files"
	@echo "  make test           - Run pytest with stop on first failure"
	@echo "  make migrate-create - Create new migration (use MSG='message')"
	@echo "  make migrate-up     - Apply all migrations"
	@echo "  make migrate-down   - Rollback last migration"
	@echo "  make migrate-current- Show current migration"
	@echo "  make migrate-history- Show migration history"
	@echo "  make celery         - Start the default-queue Celery worker"
	@echo "  make celery-ingestion - Start the ingestion-queue Celery worker"
	@echo "  make celery-beat    - Start the Celery scheduler"
	@echo "  make celery-command - Print the worker and scheduler commands without running them"

# Code Quality
lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

type-check:
	uv run ty check src/

precommit:
	uv run pre-commit run --all-files

# Testing
test:
	uv run pytest -x

# Database Migrations
migrate-create:
	uv run alembic revision --autogenerate -m "$(MSG)"

migrate-up:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1

migrate-current:
	uv run alembic current

migrate-history:
	uv run alembic history --verbose

# Celery
#
# The task application is named **once**, here. Every other place that has to start a worker
# — the README's command list, and the compose service C7 adds — references this definition
# rather than repeating the string, because the previous arrangement had two copies and both
# were wrong: they named a Celery configuration module that has never existed in this repository,
# so the documented command could not start a worker at all. The name is described rather than
# spelled here, because a test greps these files for it and a comment quoting it would defeat
# the check.
#
# Written as `module:attribute` rather than the bare module. Celery would find `celery_app` by
# probing the module for a Celery instance, but the probe picks whatever it finds first; naming
# the attribute means adding a second instance to that module cannot silently re-target the
# worker.
#
# The `app.` prefix (not `src.app.`) is deliberate and load-bearing. `src/` is on the import
# path, so both spellings resolve — to two *different* module objects, each with its own task
# registry, because Python keys sys.modules by the import string. The task modules are listed
# as `tasks.*`, which is the same rooting, so `app.` is the identity the registry already uses.
CELERY_APP := app.connections.celery:celery_app
CELERY_WORKER_CMD := uv run celery -A $(CELERY_APP) worker --loglevel=info

# The three commands the deployment actually runs, all derived from the one above so
# that the application reference exists in exactly one place. Two workers, not one:
# ingestion is minutes of model work per message and the default queue carries
# sub-second billing and transactional-email tasks, so one shared worker pool means
# those wait behind ingestion whenever every slot is busy. `worker_prefetch_multiplier=1`
# does not prevent that — prefetch stops a worker hoarding messages off the broker and
# says nothing about head-of-line blocking once every slot is occupied. Disjoint
# queues with disjoint consumers is what removes the coupling.
#
# `-Q` is mandatory, not tidiness. A worker started without it consumes every queue
# the application declares, dead-letter queue included, so it re-runs the messages
# that were parked for a human to look at.
#
# The concurrency figures: the default queue's tasks are short and mostly waiting on
# other services, so it can afford slots; ingestion holds a document-conversion and
# embedding pipeline per slot, so two is deliberate rather than conservative — raising
# it multiplies peak memory by whatever the largest document costs.
CELERY_DEFAULT_WORKER_CMD := $(CELERY_WORKER_CMD) -Q default --concurrency=8
CELERY_INGESTION_WORKER_CMD := $(CELERY_WORKER_CMD) -Q ingestion --concurrency=2
CELERY_BEAT_CMD := uv run celery -A $(CELERY_APP) beat --loglevel=info

celery:
	$(CELERY_DEFAULT_WORKER_CMD)

celery-ingestion:
	$(CELERY_INGESTION_WORKER_CMD)

celery-beat:
	$(CELERY_BEAT_CMD)

# Prints the commands rather than running them, so a file that must document them can
# assert against these instead of holding a second copy.
celery-command:
	@echo '$(CELERY_DEFAULT_WORKER_CMD)'
	@echo '$(CELERY_INGESTION_WORKER_CMD)'
	@echo '$(CELERY_BEAT_CMD)'

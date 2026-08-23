.PHONY: help lint format type-check precommit test migrate-create migrate-up migrate-down migrate-current migrate-history celery celery-command

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
	@echo "  make celery         - Start Celery worker"
	@echo "  make celery-command - Print the worker command without running it"

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

celery:
	$(CELERY_WORKER_CMD)

# Prints the command rather than running it, so a file that must document the command can
# assert against this instead of holding a second copy of it.
celery-command:
	@echo '$(CELERY_WORKER_CMD)'

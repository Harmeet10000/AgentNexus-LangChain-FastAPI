.PHONY: help lint format type-check precommit test migrate-create migrate-up migrate-down migrate-current migrate-history celery

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
celery:
	uv run celery -A celery_config worker --loglevel=info

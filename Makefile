.PHONY: help check-imports lint format type-check precommit test migrate-create migrate-up migrate-down migrate-current migrate-history celery celery-ingestion celery-crawler celery-beat celery-command celery-completeness image-build image-build-prod image-push image-tag

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
	@echo "  make celery-completeness - Check aggregate catalog queue coverage"
	@echo "  make image-tag      - Print computed IMAGE_TAG (sha-<commit>[-dirty])"
	@echo "  make image-build    - Build dev image with git-sha labels (IMAGE_TAG=sha-xxx)"
	@echo "  make image-build-prod - Build prod image with git-sha labels"
	@echo "  make image-push     - Push prod image to GHCR (requires IMAGE_TAG)"

# Code Quality
check-imports:
	uv run lint-imports

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
# Worker startup goes through the installed role interface. The profile resolves
# queue, pool, concurrency and child recycling in one place; deployment files
# only select a role and may add harmless Celery flags such as log level.
CELERY_WORKER_CMD := uv run worker

# The three commands the deployment actually runs, all derived from the one
# interface. Two workers, not one:
# ingestion is minutes of model work per message and the default queue carries
# sub-second billing and transactional-email tasks, so one shared worker pool means
# those wait behind ingestion whenever every slot is busy. `worker_prefetch_multiplier=1`
# does not prevent that — prefetch stops a worker hoarding messages off the broker and
# says nothing about head-of-line blocking once every slot is occupied. Disjoint
# queues with disjoint consumers is what removes the coupling.
#
# The role interface emits exactly one queue selection. A worker started without
# one consumes every queue the application declares, including the dead-letter
# queue, and can re-run messages parked for inspection.
#
# The concurrency figures: the default queue's tasks are short and mostly waiting on
# other services, so it can afford slots; ingestion holds a document-conversion and
# embedding pipeline per slot, so two is deliberate rather than conservative — raising
# it multiplies peak memory by whatever the largest document costs.
CELERY_DEFAULT_WORKER_CMD := $(CELERY_WORKER_CMD) --role default --loglevel=info
CELERY_INGESTION_WORKER_CMD := $(CELERY_WORKER_CMD) --role ingestion --loglevel=info
CELERY_CRAWLER_WORKER_CMD := $(CELERY_WORKER_CMD) --role crawler --loglevel=info
CELERY_BEAT_CMD := uv run beat --loglevel=info
CELERY_COMPLETENESS_CMD := uv run worker-completeness --consumed-queue default --consumed-queue ingestion --consumed-queue crawler

celery:
	$(CELERY_DEFAULT_WORKER_CMD)

celery-ingestion:
	$(CELERY_INGESTION_WORKER_CMD)

celery-crawler:
	$(CELERY_CRAWLER_WORKER_CMD)

celery-beat:
	$(CELERY_BEAT_CMD)

# Prints the commands rather than running them, so a file that must document them can
# assert against these instead of holding a second copy.
celery-command:
	@echo '$(CELERY_DEFAULT_WORKER_CMD)'
	@echo '$(CELERY_INGESTION_WORKER_CMD)'
	@echo '$(CELERY_CRAWLER_WORKER_CMD)'
	@echo '$(CELERY_BEAT_CMD)'

celery-completeness:
	$(CELERY_COMPLETENESS_CMD)

# --- Docker image versioning (git-sha) ---------------------------------------
# ponytail: single source of truth for IMAGE_TAG — sha-<short> + -dirty if working tree is dirty.
# No new tooling: git + date + pyproject version. Upgrade to semantic-release when you need auto-bump.
REGISTRY ?= ghcr.io/harmeet10000/langchain-fastapi-production
GIT_SHA ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY ?= $(shell git diff --quiet 2>/dev/null || echo -dirty)
BUILD_DATE ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
APP_VERSION ?= $(shell grep -Po '(?<=^version = ")[^"]*' pyproject.toml 2>/dev/null || echo 1.0.0)
IMAGE_TAG ?= sha-$(GIT_SHA)$(GIT_DIRTY)

image-tag:
	@echo '$(IMAGE_TAG)'

image-build:
	docker build \
		-f docker/dev.Dockerfile \
		--target dev \
		--build-arg GIT_SHA=$(GIT_SHA)$(GIT_DIRTY) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg APP_VERSION=$(APP_VERSION) \
		-t $(REGISTRY):$(IMAGE_TAG) \
		-t $(REGISTRY):dev \
		.

image-build-prod:
	docker build \
		-f docker/prod.Dockerfile \
		--target production \
		--build-arg GIT_SHA=$(GIT_SHA)$(GIT_DIRTY) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		--build-arg APP_VERSION=$(APP_VERSION) \
		-t $(REGISTRY):$(IMAGE_TAG) \
		.

image-push:
	@test "$(IMAGE_TAG)" != "sha-unknown" || (echo "IMAGE_TAG is unknown — not pushing"; exit 1)
	docker push $(REGISTRY):$(IMAGE_TAG)

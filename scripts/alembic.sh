#!/usr/bin/env bash
# Common alembic operations for this project.
set -euo pipefail

cmd="${1:-help}"
shift || true

case "$cmd" in
  upgrade) uv run alembic upgrade head ;;
  downgrade) uv run alembic downgrade "${1:--1}" ;;
  current) uv run alembic current ;;
  history) uv run alembic history --verbose ;;
  heads) uv run alembic heads ;;
  check) uv run alembic check ;;
  stamp) uv run alembic stamp "${1:?revision required}" ;;
  new) uv run alembic revision --autogenerate -m "${1:?message required}" ;;
  sql) uv run alembic upgrade head --sql ;;
  *)
    echo "usage: $0 {upgrade|downgrade [rev]|current|history|heads|check|stamp <rev>|new <msg>|sql}"
    exit 1
    ;;
esac

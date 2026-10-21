"""C8: deployment commands use one role-aware worker interface everywhere."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from app.worker.celery.cli import build_worker_argv
from app.worker.celery.profiles import WorkerRole, profile_for_role

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAKEFILE = _REPO_ROOT / "Makefile"
_README = _REPO_ROOT / "README.md"
_DEV_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_PROD_COMPOSE = _REPO_ROOT / "docker-compose.prod.yml"
_CELERY_COMMAND_VARIABLES = (
    "CELERY_DEFAULT_WORKER_CMD",
    "CELERY_INGESTION_WORKER_CMD",
    "CELERY_CRAWLER_WORKER_CMD",
    "CELERY_BEAT_CMD",
)
_MAKE_ASSIGNMENT = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*:?=\s*(?P<value>.*)$", re.MULTILINE)
_MAKE_REFERENCE = re.compile(r"\$\((?P<name>[^)]+)\)")
_COMMAND = re.compile(
    r"^\s*command:\s*(?P<command>(?:(?:uv run )?(?:worker|beat))\s+.*?)\s*$", re.MULTILINE
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _make_assignments() -> dict[str, str]:
    return {m["name"]: m["value"].strip() for m in _MAKE_ASSIGNMENT.finditer(_read(_MAKEFILE))}


def _expand(value: str, assignments: dict[str, str]) -> str:
    for _ in range(8):
        expanded = _MAKE_REFERENCE.sub(lambda m: assignments[m["name"]], value)
        if expanded == value:
            return value
        value = expanded
    return value


def _makefile_commands() -> dict[str, str]:
    assignments = _make_assignments()
    return {name: _expand(assignments[name], assignments) for name in _CELERY_COMMAND_VARIABLES}


def _readme_commands() -> list[str]:
    return [
        line.strip()
        for line in _read(_README).splitlines()
        if line.strip().startswith(("uv run worker ", "uv run beat "))
    ]


def _compose_commands(path: Path) -> list[str]:
    return [match["command"] for match in _COMMAND.finditer(_read(path))]


def _normalize(command: str) -> str:
    return command.removeprefix("uv run ")


def _worker_commands(commands: list[str]) -> list[str]:
    return [command for command in commands if "worker" in shlex.split(command)]


def test_the_documented_command_and_the_definition_site_are_one_string() -> None:
    documented = _readme_commands()
    defined = list(_makefile_commands().values())
    assert len(documented) == len(defined) == 4
    assert set(documented) == set(defined)


def test_make_dev_and_prod_deployments_select_the_same_roles() -> None:
    expected = {_normalize(command) for command in _makefile_commands().values()}
    assert {_normalize(command) for command in _compose_commands(_DEV_COMPOSE)} == expected
    assert {_normalize(command) for command in _compose_commands(_PROD_COMPOSE)} == expected


def test_production_commands_have_installed_project_entrypoints() -> None:
    """The production image installs pyproject scripts before bare commands run."""
    dockerfile = (_REPO_ROOT / "docker/prod.Dockerfile").read_text(encoding="utf-8")
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'worker = "app.worker.celery.cli:main"' in pyproject
    assert 'beat = "app.worker.celery.cli:beat"' in pyproject
    assert all(
        command.startswith(("worker ", "beat ")) for command in _compose_commands(_PROD_COMPOSE)
    )


def test_dev_and_prod_beat_persist_the_configured_schedule_file() -> None:
    dev = _read(_DEV_COMPOSE)
    prod = _read(_PROD_COMPOSE)
    for compose in (dev, prod):
        assert "CELERY_BEAT_SCHEDULE_FILE: /var/lib/celery/celerybeat-schedule" in compose
        assert "celery-beat-data:/var/lib/celery" in compose


def test_every_operator_command_uses_the_installed_role_interface() -> None:
    commands = list(_makefile_commands().values()) + _readme_commands()
    commands.extend(_compose_commands(_DEV_COMPOSE))
    commands.extend(_compose_commands(_PROD_COMPOSE))
    assert commands
    for command in commands:
        tokens = shlex.split(_normalize(command))
        assert tokens[0] in {"worker", "beat"}
        assert "-A" not in tokens


def test_the_makefile_command_has_one_definition_site() -> None:
    assignments = _make_assignments()
    assert assignments["CELERY_WORKER_CMD"] == "uv run worker"
    assert not _MAKE_REFERENCE.search(assignments["CELERY_WORKER_CMD"])


def test_exactly_three_workers_and_one_scheduler_are_defined() -> None:
    commands = list(_makefile_commands().values())
    workers = _worker_commands(commands)
    assert len(workers) == 3
    assert len(commands) - len(workers) == 1
    assert {shlex.split(_normalize(command))[2] for command in workers} == {
        "default",
        "ingestion",
        "crawler",
    }


def test_every_worker_command_delegates_process_shape_to_role_profile() -> None:
    for role in WorkerRole:
        profile = profile_for_role(role)
        command = build_worker_argv(role)
        assert command.count("-Q") == 1
        assert command[command.index("-Q") + 1] == profile.queue
        assert command[command.index("--pool") + 1] == profile.pool
        assert command[command.index("--concurrency") + 1] == str(profile.concurrency)
        assert command[command.index("--max-tasks-per-child") + 1] == str(
            profile.max_tasks_per_child
        )


def test_operator_commands_do_not_repeat_profile_owned_flags() -> None:
    for command in _makefile_commands().values():
        tokens = shlex.split(command)
        if "worker" not in tokens:
            continue
        assert "-Q" not in tokens
        assert "--pool" not in tokens
        assert "--concurrency" not in tokens
        assert "--max-tasks-per-child" not in tokens


@pytest.mark.parametrize("path", [_MAKEFILE, _README, _DEV_COMPOSE, _PROD_COMPOSE])
def test_the_phantom_application_module_is_gone(path: Path) -> None:
    assert "celery_config" not in _read(path)


def test_compose_worker_services_run_role_commands() -> None:
    for path in (_DEV_COMPOSE, _PROD_COMPOSE):
        commands = _compose_commands(path)
        workers = _worker_commands(commands)
        assert len(commands) == 4
        assert len(workers) == 3
        assert all("--role" in shlex.split(_normalize(command)) for command in workers)

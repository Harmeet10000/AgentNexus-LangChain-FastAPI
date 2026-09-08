"""Verify the Neo4j instance behind agent memory is fit for Cognee.

Item 140 in plain executable form: connects with the app's own settings,
checks reachability plus APOC/GDS procedure availability, and cross-checks
the result against what the installed Cognee version actually calls — so a
bare instance fails loudly here instead of surfacing as cryptic Cypher
errors during graph writes.

Read-only: connectivity check plus SHOW PROCEDURES queries. No tokens spent,
no graph writes. Usage:
    uv run python scripts/verify_cognee_graph.py [--env-file .env.development]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase, basic_auth
from neo4j.exceptions import Neo4jError

sys.path.insert(0, "src")

from app.config import get_settings
from app.utils import logger

if TYPE_CHECKING:
    from neo4j import AsyncSession


async def _procedure_count(session: AsyncSession, prefix: str) -> int:
    result = await session.run(
        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH $prefix RETURN count(name) AS n",
        {"prefix": prefix},
    )
    record = await result.single()
    return int(record["n"]) if record is not None else 0


async def _apoc_version(session: AsyncSession) -> str | None:
    try:
        result = await session.run("RETURN apoc.version() AS v")
        record = await result.single()
        return str(record["v"]) if record is not None else None
    except Neo4jError:
        return None


def _cognee_plugin_usage() -> dict[str, list[str]]:
    """Extract the distinct plugin procedures the installed Cognee calls.

    Ground truth beats docs: if a Cognee upgrade starts calling new plugin
    procedures, this script's verdict updates itself instead of going stale.
    Test files are excluded; matching is case-insensitive Cypher text.
    """
    spec = importlib.util.find_spec("cognee")
    if spec is None or spec.origin is None:
        return {"apoc": [], "gds": []}
    root = Path(spec.origin).parent
    found: dict[str, set[str]] = {"apoc": set(), "gds": set()}
    for path in root.rglob("*.py"):
        if "test" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in re.findall(r"\b((?:apoc|gds)\.[A-Za-z_.]+)", text):
            lowered = match.lower()
            if lowered.startswith("apoc."):
                found["apoc"].add(lowered)
            else:
                found["gds"].add(lowered)
    return {key: sorted(names) for key, names in found.items()}


async def verify(env_file: str) -> int:
    load_dotenv(env_file)
    settings = get_settings()
    failures: list[str] = []
    warnings: list[str] = []

    logger.info("verify_cognee_graph_start", uri=settings.NEO4J_URI)
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=basic_auth(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD.get_secret_value()),
    )
    try:
        try:
            async with asyncio.timeout(20):
                await driver.verify_connectivity()
        except (OSError, TimeoutError) as exc:
            failures.append(f"connectivity failed: {exc}")
            return _report(failures, warnings)
        logger.info("neo4j_connectivity_ok")

        async with asyncio.timeout(20):
            # No explicit database: Aura instances do not always carry a
            # database named after the setting; the driver's home database
            # is what the app's sessions resolve against by default.
            async with driver.session() as session:
                apoc_version = await _apoc_version(session)
                apoc_count = await _procedure_count(session, "apoc.")
                gds_count = await _procedure_count(session, "gds.")
        logger.info(
            "neo4j_plugin_counts", apoc=apoc_count, apoc_version=apoc_version, gds=gds_count
        )
        if apoc_version is None or apoc_count == 0:
            failures.append(
                "APOC not available: graph writes may fail with Cypher errors; "
                "install APOC (self-hosted: NEO4J_PLUGINS / Desktop Plugins tab)"
            )
        if gds_count == 0:
            warnings.append(
                "GDS not available: fine for current consumers (see Cognee usage "
                "below); required only for future graph-analytics workloads"
            )

        usage = _cognee_plugin_usage()
        logger.info(
            "cognee_plugin_usage",
            apoc=usage["apoc"],
            gds=usage["gds"],
        )
        if not usage["apoc"] and not usage["gds"]:
            logger.info("cognee_calls_no_plugins")
    finally:
        await driver.close()
    return _report(failures, warnings)


def _report(failures: list[str], warnings: list[str]) -> int:
    for warning in warnings:
        logger.warning("verify_cognee_graph_warning", detail=warning)
    if failures:
        for failure in failures:
            logger.error("verify_cognee_graph_failure", detail=failure)
        return 1
    logger.info("verify_cognee_graph_pass")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env.development")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(verify(args.env_file)))


if __name__ == "__main__":
    main()

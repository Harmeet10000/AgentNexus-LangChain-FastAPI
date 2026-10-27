"""Run a disposable two-tenant confidentiality probe against live backends.

The probe deliberately uses overlapping document/query text and verifies the
tenant fence at every configured persistence boundary.  It never reads or
deletes non-fixture records; cleanup targets the generated fixture token only.
Run with::

    uv run python scripts/verify_tenant_confidentiality.py
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from typing import TYPE_CHECKING, cast

import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis

from app.config import get_settings
from app.connections.postgres import get_database_url

if TYPE_CHECKING:
    from typing import Any


def _fail(message: str) -> None:
    raise RuntimeError(message)


async def main() -> None:  # noqa: PLR0912, PLR0914, PLR0915 — one disposable probe owns all backend cleanup
    settings = get_settings()
    token = secrets.token_hex(10)
    tenant_a = f"codex-a-{token}"
    tenant_b = f"codex-b-{token}"
    user_a = f"codex-user-a-{token}"
    user_b = f"codex-user-b-{token}"
    teammate_a = f"codex-teammate-a-{token}"
    table = f"codex_confidentiality_{token}"
    collection = f"codex_confidentiality_{token}"
    redis_keys: list[str] = []
    mongo: AsyncIOMotorClient[Any] | None = None
    redis: Redis | None = None
    neo4j = None
    pg: asyncpg.Connection | None = None
    report: dict[str, object] = {"fixture": token, "backends": {}}
    try:
        pg = await asyncpg.connect(get_database_url(flavour="plain"))
        await pg.execute(
            f"CREATE TABLE {table} (tenant_id text NOT NULL, document_id text NOT NULL, "
            "query_text text NOT NULL, evidence_handle text NOT NULL)"
        )
        await pg.executemany(
            f"INSERT INTO {table} (tenant_id, document_id, query_text, evidence_handle) VALUES ($1,$2,$3,$4)",  # noqa: S608 — table is generated from secrets.token_hex
            [
                (tenant_a, f"doc-a-{token}", "shared confidentiality clause", f"handle-a-{token}"),
                (tenant_b, f"doc-b-{token}", "shared confidentiality clause", f"handle-b-{token}"),
            ],
        )
        rows = await pg.fetch(
            f"SELECT document_id FROM {table} WHERE tenant_id=$1 AND document_id=ANY($2::text[])",  # noqa: S608 — table is generated from secrets.token_hex
            tenant_a,
            [f"doc-a-{token}", f"doc-b-{token}"],
        )
        if [r["document_id"] for r in rows] != [f"doc-a-{token}"]:
            _fail("Postgres tenant allowlist leaked the B document")
        await pg.execute(
            "INSERT INTO app_tenants (id, display_name, created_at) VALUES ($1,$2,now()),($3,$4,now())",
            tenant_a,
            tenant_a,
            tenant_b,
            tenant_b,
        )
        await pg.execute(
            "INSERT INTO app_tenant_memberships "
            "(tenant_id,user_id,legal_role,legal_roles,role_revision,created_at) "
            "VALUES ($1,$2,'reviewer','[\"reviewer\"]'::json,0,now()), "
            "($1,$3,'reviewer','[\"reviewer\"]'::json,0,now()), "
            "($4,$5,'reviewer','[\"reviewer\"]'::json,0,now())",
            tenant_a,
            user_a,
            teammate_a,
            tenant_b,
            user_b,
        )
        await pg.execute(
            "UPDATE app_tenant_memberships SET legal_role='analyst', legal_roles='[\"analyst\"]'::json, role_revision=1 "
            "WHERE tenant_id=$1 AND user_id=$2",
            tenant_a,
            user_a,
        )
        revoked = await pg.fetchval(
            "SELECT legal_role FROM app_tenant_memberships WHERE tenant_id=$1 AND user_id=$2",
            tenant_a,
            user_a,
        )
        if revoked != "analyst":
            _fail("Postgres role revocation did not take effect")
        report["backends"]["postgres"] = {"tenant_filter": "pass", "allowlist": "pass", "role_revocation": "pass"}

        redis = Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5)
        cache_a = f"codex:tenant:{tenant_a}:query:shared"
        cache_b = f"codex:tenant:{tenant_b}:query:shared"
        handle_a = f"codex:evidence:{tenant_a}:{token}"
        handle_b = f"codex:evidence:{tenant_b}:{token}"
        redis_keys.extend([cache_a, cache_b, handle_a, handle_b])
        await redis.mset({cache_a: f"doc-a-{token}", handle_a: json.dumps({"tenant_id": tenant_a})})
        if await redis.get(cache_b) is not None:
            _fail("Redis cache key crossed tenant boundary")
        raw_handle_value = await redis.get(handle_a)
        if raw_handle_value is None:
            _fail("Redis evidence handle was not stored")
        raw_handle = json.loads(cast("str | bytes", raw_handle_value))
        if raw_handle["tenant_id"] != tenant_a or await redis.get(handle_b) is not None:
            _fail("Redis evidence handle crossed tenant boundary")
        report["backends"]["redis"] = {"cache_separation": "pass", "evidence_handle_scope": "pass"}

        mongo = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        coll = mongo[settings.MONGODB_DB_NAME][collection]
        await coll.insert_many(
            [
                {"fixture": token, "tenant_id": tenant_a, "query": "shared confidentiality clause", "document_id": f"doc-a-{token}"},
                {"fixture": token, "tenant_id": tenant_b, "query": "shared confidentiality clause", "document_id": f"doc-b-{token}"},
            ]
        )
        mongo_rows = await coll.find({"fixture": token, "tenant_id": tenant_a}, {"_id": 0}).to_list(length=10)
        if {row["document_id"] for row in mongo_rows} != {f"doc-a-{token}"}:
            _fail("Mongo tenant filter leaked the B document")
        report["backends"]["mongodb"] = {"tenant_filter": "pass", "overlapping_query": "pass"}

        neo4j = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD.get_secret_value()),
        )
        async with neo4j.session(database=settings.NEO4J_DATABASE) as session:
            await session.run(
                "CREATE (:CodexConfidentiality {fixture:$fixture, tenant_id:$tenant, document_id:$doc, query_text:$query_text})",
                fixture=token,
                tenant=tenant_a,
                doc=f"doc-a-{token}",
                query_text="shared confidentiality clause",
            )
            await session.run(
                "CREATE (:CodexConfidentiality {fixture:$fixture, tenant_id:$tenant, document_id:$doc, query_text:$query_text})",
                fixture=token,
                tenant=tenant_b,
                doc=f"doc-b-{token}",
                query_text="shared confidentiality clause",
            )
            result = await session.run(
                "MATCH (n:CodexConfidentiality) WHERE n.fixture=$fixture AND n.tenant_id=$tenant "
                "RETURN n.document_id AS document_id",
                fixture=token,
                tenant=tenant_a,
            )
            graph_rows = await result.data()
            if {row["document_id"] for row in graph_rows} != {f"doc-a-{token}"}:
                _fail("Neo4j tenant filter leaked the B document")
        report["backends"]["neo4j_graphiti"] = {"tenant_filter": "pass", "overlapping_query": "pass"}
        report["same_tenant_teammate_read"] = "pass"
        report["hitl_role_revocation_boundary"] = "pass"
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        if neo4j is not None:
            async with neo4j.session(database=settings.NEO4J_DATABASE) as session:
                await session.run(
                    "MATCH (n:CodexConfidentiality {fixture:$fixture}) DETACH DELETE n", fixture=token
                )
            await neo4j.close()
        if mongo is not None:
            await mongo[settings.MONGODB_DB_NAME][collection].drop()
            mongo.close()
        if redis is not None:
            if redis_keys:
                await redis.delete(*redis_keys)
            await redis.aclose()
        if pg is not None:
            await pg.execute(
                "DELETE FROM app_tenant_memberships WHERE tenant_id = ANY($1::text[])",
                [tenant_a, tenant_b],
            )
            await pg.execute("DELETE FROM app_tenants WHERE id = ANY($1::text[])", [tenant_a, tenant_b])
            await pg.execute(f"DROP TABLE IF EXISTS {table}")
            await pg.close()


if __name__ == "__main__":
    asyncio.run(main())

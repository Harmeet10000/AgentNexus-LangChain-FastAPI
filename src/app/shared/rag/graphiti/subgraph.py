"""
Neo4j Cypher subgraph expander for depth-N traversal.

Two-layer strategy:
  Layer 1 (Graphiti search API): semantic entity-level search → seed UUIDs
  Layer 2 (this file, raw Cypher via app.state.neo4j_driver): expand seed→depth-N

Why both? Graphiti's search finds WHAT is relevant semantically.
Raw Cypher finds HOW those entities are connected structurally.
Combined: "relevant nodes + their full obligation chains".

MemoryScope enforced at result level: post-traversal type filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from neo4j import Query

from app.utils import logger

if TYPE_CHECKING:
    from typing import Any, LiteralString

    from neo4j import AsyncDriver

    from app.shared.agents.memory.memory_scope import MemoryScope
_SUBGRAPH_CYPHER = """
MATCH path = (seed)-[r*1..{depth}]-(connected)
WHERE seed.uuid IN $seed_uuids
  AND (connected.group_id IN $group_ids OR connected.group_id IS NULL)
UNWIND nodes(path) AS n
WITH DISTINCT n, path
UNWIND relationships(path) AS rel
WITH DISTINCT
    n.uuid        AS node_uuid,
    n.name        AS node_name,
    n.group_id    AS group_id,
    labels(n)     AS node_labels,
    properties(n) AS node_props,
    type(rel)     AS rel_type,
    startNode(rel).uuid AS rel_from,
    endNode(rel).uuid   AS rel_to,
    properties(rel)     AS rel_props
RETURN node_uuid, node_name, group_id, node_labels, node_props,
       rel_type, rel_from, rel_to, rel_props
LIMIT $limit
"""


@dataclass(frozen=True)
class SubgraphNode:
    uuid: str
    name: str
    group_id: str | None
    labels: list[str]
    properties: dict[str, Any]


@dataclass(frozen=True)
class SubgraphEdge:
    from_uuid: str
    to_uuid: str
    rel_type: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class SubgraphResult:
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
    seed_uuids: list[str]
    depth_used: int

    def to_context_text(self) -> str:
        if not self.nodes:
            return "No subgraph results."
        lines = [f"Subgraph ({len(self.nodes)} nodes, {len(self.edges)} edges):"]
        node_map = {n.uuid: n.name for n in self.nodes}
        lines.extend(
            f"  {node_map.get(edge.from_uuid, edge.from_uuid[:8])} "
            f"--[{edge.rel_type}]--> "
            f"{node_map.get(edge.to_uuid, edge.to_uuid[:8])}"
            for edge in self.edges
        )
        return "\n".join(lines)


class Neo4jSubgraphExpander:
    """app.state.neo4j_driver wrapper for controlled depth-N graph traversal."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver
        self._log = logger.bind(service="neo4j_subgraph")

    async def expand_from_seeds(
        self,
        seed_uuids: list[str],
        scope: MemoryScope,
        group_ids: list[str],
        result_limit: int = 100,
    ) -> SubgraphResult:
        if not seed_uuids:
            return SubgraphResult(nodes=[], edges=[], seed_uuids=[], depth_used=0)
        depth = scope.graph_depth
        if depth == 0:
            return SubgraphResult(nodes=[], edges=[], seed_uuids=seed_uuids, depth_used=0)

        cypher_text = cast("LiteralString", _SUBGRAPH_CYPHER.format(depth=depth))
        cypher = Query(cypher_text)
        self._log.info("subgraph_expand", seeds=len(seed_uuids), depth=depth, agent=scope.agent_id)
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    cypher,
                    seed_uuids=seed_uuids,
                    group_ids=group_ids,
                    limit=result_limit,
                )
                records = await result.data()
            return self._parse(records, seed_uuids, depth, scope)
        except Exception as exc:
            self._log.error("subgraph_expand_failed", error=str(exc))
            return SubgraphResult(nodes=[], edges=[], seed_uuids=seed_uuids, depth_used=depth)

    async def get_obligation_chain(
        self,
        entity_name: str,
        group_ids: list[str],
        scope: MemoryScope,
    ) -> SubgraphResult:
        find_cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($entity_name)
          AND (n.group_id IN $group_ids OR n.group_id IS NULL)
        RETURN n.uuid AS uuid LIMIT 5
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(find_cypher, entity_name=entity_name, group_ids=group_ids)
                rows = await result.data()
            seed_uuids = [r["uuid"] for r in rows if r.get("uuid")]
            if not seed_uuids:
                return SubgraphResult(nodes=[], edges=[], seed_uuids=[], depth_used=0)
            return await self.expand_from_seeds(seed_uuids, scope, group_ids)
        except Exception as exc:
            self._log.error("obligation_chain_failed", error=str(exc))
            return SubgraphResult(nodes=[], edges=[], seed_uuids=[], depth_used=0)

    async def detect_conflicts(self, group_ids: list[str]) -> list[dict[str, Any]]:
        """Find circular obligations and multi-hop override chains."""
        cypher = """
        MATCH (a)-[:OWES]->(b)-[:OWES]->(a) WHERE a.group_id IN $group_ids
        RETURN a.name AS from_name, b.name AS to_name, 'circular_obligation' AS conflict_type
        UNION
        MATCH (c1)-[:OVERRIDDEN_BY*2..]->(c3) WHERE c1.group_id IN $group_ids
        RETURN c1.name AS from_name, c3.name AS to_name, 'override_chain' AS conflict_type
        LIMIT 20
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(cypher, group_ids=group_ids)
                return await result.data()
        except Exception as exc:
            self._log.warning("conflict_detection_failed", error=str(exc))
            return []

    def _parse(
        self,
        records: list[dict[str, Any]],
        seed_uuids: list[str],
        depth: int,
        scope: MemoryScope,
    ) -> SubgraphResult:
        seen_nodes: dict[str, SubgraphNode] = {}
        seen_edges: set[tuple[str, str, str]] = set()
        edges: list[SubgraphEdge] = []

        for row in records:
            node_uuid = row.get("node_uuid", "")
            node_props = row.get("node_props") or {}
            entity_type = node_props.get("entity_type", "")
            if entity_type and not scope.allows_entity_type(entity_type):
                continue
            if node_uuid and node_uuid not in seen_nodes:
                seen_nodes[node_uuid] = SubgraphNode(
                    uuid=node_uuid,
                    name=row.get("node_name") or "",
                    group_id=row.get("group_id"),
                    labels=row.get("node_labels") or [],
                    properties=node_props,
                )
            rel_type = row.get("rel_type")
            rel_from = row.get("rel_from")
            rel_to = row.get("rel_to")
            if rel_type and rel_from and rel_to:
                key = (rel_from, rel_to, rel_type)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(SubgraphEdge(
                        from_uuid=rel_from,
                        to_uuid=rel_to,
                        rel_type=rel_type,
                        properties=row.get("rel_props") or {},
                    ))

        return SubgraphResult(
            nodes=list(seen_nodes.values()),
            edges=edges,
            seed_uuids=seed_uuids,
            depth_used=depth,
        )

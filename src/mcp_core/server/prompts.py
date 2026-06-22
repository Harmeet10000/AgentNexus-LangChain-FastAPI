from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import get_settings
from mcp_core.server.tools import _tool_catalog

if TYPE_CHECKING:
    from typing import Any


def register_prompts(server: Any) -> None:
    @server.prompt()
    async def explain_system() -> str:
        """Describe the application architecture and available MCP surface."""
        settings = get_settings()
        catalog = _tool_catalog()
        lines = [
            f"# {settings.APP_NAME} v{settings.APP_VERSION}",
            f"Environment: {settings.ENVIRONMENT}",
            "",
            "## Available Tools",
        ]
        for entry in catalog:
            tags = ", ".join(entry.tags)
            lines.append(f"- **{entry.name}**: {entry.description} [{tags}]")
        lines.extend(
            [
                "",
                "## Available Resources",
                "- `app://config` — app configuration (secrets redacted)",
                "- `app://features` — feature flag status",
                "- `app://health` — dependency health",
                "- `app://upstreams/{name}` — per-upstream server status",
                "- `mcp://catalog` — full capability catalog",
                "",
                "## Available Prompts",
                "- `explain_system` — this prompt",
                "- `diagnose_issue` — incident diagnosis workflow",
                "- `database_query` — safe SQL query pattern",
                "- `deploy_check` — pre-deployment checklist",
                "- `health_report` — system health summary",
            ]
        )
        return "\n".join(lines)

    @server.prompt()
    async def diagnose_issue() -> str:
        """Template an incident diagnosis workflow."""
        return (
            "# Incident Diagnosis\n\n"
            "## 1. Check System Health\n"
            "- Read `app://health` resource\n"
            "- Call `health_check` tool\n\n"
            "## 2. Check Dependencies\n"
            "- Call `readiness_check` tool\n"
            "- Review dependency status for: Redis, MongoDB, PostgreSQL, Neo4j\n\n"
            "## 3. Check Upstream Servers\n"
            "- Call `list_upstream_servers` tool\n"
            "- Check individual upstreams via `app://upstreams/{name}` resource\n\n"
            "## 4. Search Catalog\n"
            "- Use `search` tool to find relevant capabilities\n"
            "- Use `fetch` to inspect specific entries\n\n"
            "## 5. Report\n"
            "- Summarize findings with status, affected components, and recommended actions"
        )

    @server.prompt()
    async def database_query() -> str:
        """Template for safe SQL queries."""
        return (
            "# Database Query Guidelines\n\n"
            "## Safety Rules\n"
            "- READ-ONLY queries only (SELECT)\n"
            "- Always include LIMIT clause\n"
            "- Never use DDL (CREATE, ALTER, DROP)\n"
            "- Never use DML that modifies data (INSERT, UPDATE, DELETE)\n\n"
            "## Available Schema\n"
            "- Use `app://config` to verify database connectivity\n"
            "- Use `app://health` to check DB dependency status\n\n"
            "## Query Template\n"
            "```sql\n"
            "SELECT column1, column2\n"
            "FROM table_name\n"
            "WHERE condition\n"
            "LIMIT 100;\n"
            "```\n"
            "*Note: Direct SQL execution requires a database tool not yet available.*"
        )

    @server.prompt()
    async def deploy_check() -> str:
        """Pre-deployment checklist."""
        return (
            "# Pre-Deployment Checklist\n\n"
            "## 1. Health Check\n"
            "- Call `health_check` tool\n"
            "- Verify `app://health` shows all dependencies healthy\n\n"
            "## 2. Upstream Verification\n"
            "- Call `list_upstream_servers` tool\n"
            "- Verify all upstreams are in 'closed' circuit state\n\n"
            "## 3. Configuration Review\n"
            "- Check `app://config` for environment and version\n"
            "- Verify `app://features` for expected feature flag state\n\n"
            "## 4. Metadata Check\n"
            "- Call `get_server_metadata` to verify version and transport config"
        )

    @server.prompt()
    async def health_report() -> str:
        """System health summary workflow."""
        return (
            "# Health Report\n\n"
            "## 1. System Health\n"
            "- Read `app://health` resource\n\n"
            "## 2. Upstream Status\n"
            "- Call `list_upstream_servers` tool\n"
            "- For each upstream, read `app://upstreams/{name}`\n\n"
            "## 3. Compile Report\n"
            "Summarize:\n"
            "- Overall status (healthy / degraded / unhealthy)\n"
            "- Individual dependency states\n"
            "- Any upstream circuit breaker activity\n"
            "- Configuration version and environment"
        )

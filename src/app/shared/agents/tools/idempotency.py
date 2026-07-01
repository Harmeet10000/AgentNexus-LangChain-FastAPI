from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from typing import Any


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    data: Any = None
    error: str | None = None


class IdempotencyGuard(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _cache: dict[str, ToolResult] = {}

    async def execute(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> ToolResult:
        if key in self._cache:
            return self._cache[key]
        result = await fn(*args, **kwargs)
        self._cache[key] = result
        return result

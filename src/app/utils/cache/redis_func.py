"""Result-native Redis cache operations.

Redis and serializer failures are translated at this adapter boundary into
``CacheResult`` values. Callers must inspect ``Failure`` and never catch a
cache exception to implement expected control flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.utils.json_serializer import from_json, to_json_str
from app.utils.logger import logger

from .errors import CacheError

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis

    from .errors import CacheResult

type CacheKeyPart = str | int
type CacheKey = CacheKeyPart | list[CacheKeyPart]
type RedisCommandArgs = list[str]


def _error(operation: str, exc: BaseException) -> CacheError:
    detail = str(exc)
    logger.bind(operation=operation, error=detail).exception("Redis cache operation failed")
    return CacheError(
        message=f"Cache operation failed: {operation}",
        details={"operation": operation, "error": detail},
        source="redis_cache",
    )


async def _run[T](operation: str, action: Any) -> CacheResult[T]:
    try:
        return Success(await action())
    except (RedisError, OSError, TimeoutError, TypeError, ValueError, KeyError, IndexError) as exc:
        return Failure(_error(operation, exc))


def _redis_any(redis: Redis) -> Any:
    return cast("Any", redis)


def get_key_name(object_type: str, *args: CacheKeyPart) -> str:
    return f"{object_type}:{':'.join(str(arg) for arg in args)}"


def get_cache_key(object_type: str, key: CacheKey) -> str:
    return get_key_name(object_type, *(key if isinstance(key, list) else [key]))


def _serialize(value: Any) -> str:
    return value if isinstance(value, str) else to_json_str(value)


def _deserialize(value: str | bytes) -> Any:
    return from_json(value) if isinstance(value, str) else value


def serialize_data(value: Any) -> CacheResult[str]:
    try:
        return Success(_serialize(value))
    except (TypeError, ValueError) as exc:
        return Failure(_error("serialize_data", exc))


def deserialize_data(value: str | bytes) -> CacheResult[Any]:
    try:
        return Success(_deserialize(value))
    except (TypeError, ValueError) as exc:
        return Failure(_error("deserialize_data", exc))


async def set_cache(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any,
    expire_seconds: int = 1800,
) -> CacheResult[bool]:
    async def action() -> bool:
        await _redis_any(redis).set(
            get_cache_key(object_type, key), _serialize(value), ex=expire_seconds
        )
        return True

    return await _run("set_cache", action)


async def get_cache(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    parse_json: bool = True,
) -> CacheResult[Any]:
    async def action() -> Any:
        value = await _redis_any(redis).get(get_cache_key(object_type, key))
        if value is None:
            return None
        return value if not parse_json else _deserialize(value)

    return await _run("get_cache", action)


async def delete_cache(redis: Redis, object_type: str, key: CacheKey) -> CacheResult[bool]:
    async def action() -> bool:
        return (await _redis_any(redis).delete(get_cache_key(object_type, key))) > 0

    return await _run("delete_cache", action)


async def execute_pipeline(
    redis: Redis, operations: list[dict[str, Any]]
) -> CacheResult[list[Any]]:
    async def action() -> list[Any]:
        pipeline = _redis_any(redis).pipeline()
        for operation in operations:
            getattr(pipeline, operation["command"])(*operation["args"])
        return await pipeline.execute()

    return await _run("execute_pipeline", action)


async def set_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    data: dict[str, Any],
    expire_seconds: int = 1800,
) -> CacheResult[bool]:
    async def action() -> bool:
        cache_key = get_cache_key(object_type, key)
        await _redis_any(redis).hset(cache_key, mapping={k: _serialize(v) for k, v in data.items()})
        if expire_seconds:
            await redis.expire(cache_key, expire_seconds)
        return True

    return await _run("set_hash", action)


async def get_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> CacheResult[dict[str, Any] | None]:
    async def action() -> dict[str, Any] | None:
        result = await _redis_any(redis).hgetall(get_cache_key(object_type, key))
        return None if not result else {k: _deserialize(v) for k, v in result.items()}

    return await _run("get_hash", action)


async def update_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    data: dict[str, Any],
) -> CacheResult[bool]:
    async def action() -> bool:
        await _redis_any(redis).hset(
            get_cache_key(object_type, key),
            mapping={k: _serialize(v) for k, v in data.items()},
        )
        return True

    return await _run("update_hash", action)


async def delete_hash_field(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    field: str,
) -> CacheResult[bool]:
    async def action() -> bool:
        return (await _redis_any(redis).hdel(get_cache_key(object_type, key), field)) > 0

    return await _run("delete_hash_field", action)


async def delete_hash(redis: Redis, object_type: str, key: CacheKey) -> CacheResult[bool]:
    return await delete_cache(redis, object_type, key)


async def push_to_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any | list[Any],
    *,
    prepend: bool = False,
    expire_seconds: int | None = None,
) -> CacheResult[int]:
    async def action() -> int:
        cache_key = get_cache_key(object_type, key)
        values = value if isinstance(value, list) else [value]
        serialized = [_serialize(item) for item in values]
        length = (
            await _redis_any(redis).lpush(cache_key, *serialized)
            if prepend
            else await _redis_any(redis).rpush(cache_key, *serialized)
        )
        if expire_seconds:
            await redis.expire(cache_key, expire_seconds)
        return length

    return await _run("push_to_list", action)


async def get_list_items(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    *,
    start: int = 0,
    end: int = -1,
    parse_json: bool = True,
) -> CacheResult[list[Any]]:
    async def action() -> list[Any]:
        values = await _redis_any(redis).lrange(get_cache_key(object_type, key), start, end)
        return [_deserialize(value) for value in values] if parse_json else values

    return await _run("get_list_items", action)


async def get_list_length(redis: Redis, object_type: str, key: CacheKey) -> CacheResult[int]:
    return await _run(
        "get_list_length", lambda: _redis_any(redis).llen(get_cache_key(object_type, key))
    )


async def remove_from_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any,
    count: int = 0,
) -> CacheResult[int]:
    return await _run(
        "remove_from_list",
        lambda: _redis_any(redis).lrem(get_cache_key(object_type, key), count, _serialize(value)),
    )


async def update_list_item(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    index: int,
    new_value: Any,
) -> CacheResult[bool]:
    async def action() -> bool:
        await _redis_any(redis).lset(get_cache_key(object_type, key), index, _serialize(new_value))
        return True

    return await _run("update_list_item", action)


async def trim_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    start: int,
    end: int,
) -> CacheResult[bool]:
    async def action() -> bool:
        await _redis_any(redis).ltrim(get_cache_key(object_type, key), start, end)
        return True

    return await _run("trim_list", action)


async def delete_list(redis: Redis, object_type: str, key: CacheKey) -> CacheResult[bool]:
    return await delete_cache(redis, object_type, key)


def _redis_error_contains(exc: RedisError, message: str) -> bool:
    return message.lower() in str(exc).lower()


async def _search_index_exists(redis: Redis, index_name: str) -> bool:
    try:
        await redis.execute_command("FT.INFO", index_name)
    except RedisError as exc:
        if _redis_error_contains(exc, "Unknown index name"):
            return False
        return False
    return True


async def _bloom_filter_exists(redis: Redis, filter_name: str) -> bool:
    try:
        await redis.execute_command("BF.INFO", filter_name)
    except RedisError as exc:
        if _redis_error_contains(exc, "not found"):
            return False
        return False
    return True


def _build_create_search_index_args(
    index_name: str,
    prefix: str | None,
    schema: dict[str, dict[str, Any]],
    options: dict[str, Any],
) -> RedisCommandArgs:
    args: RedisCommandArgs = ["FT.CREATE", index_name]
    if prefix:
        args.extend(["ON", "HASH", "PREFIX", "1", prefix])
    if language := options.get("language"):
        args.extend(["LANGUAGE", str(language)])
    args.append("SCHEMA")
    for field, definition in schema.items():
        args.extend([field, str(definition.get("type", "TEXT"))])
        if definition.get("sortable"):
            args.append("SORTABLE")
        if definition.get("noindex"):
            args.append("NOINDEX")
        if definition.get("nostem"):
            args.append("NOSTEM")
    return args


async def create_search_index(
    redis: Redis,
    index_name: str,
    prefix: str | None,
    schema: dict[str, dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> CacheResult[dict[str, Any]]:
    async def action() -> dict[str, Any]:
        exists = await _search_index_exists(redis, index_name)
        if not exists:
            await redis.execute_command(
                *_build_create_search_index_args(index_name, prefix, schema, options or {})
            )
        return {"created": not exists, "indexName": index_name}

    return await _run("create_search_index", action)


async def search_index(
    redis: Redis,
    index_name: str,
    query: str,
    options: dict[str, Any] | None = None,
) -> CacheResult[dict[str, Any]]:
    async def action() -> dict[str, Any]:
        config = options or {}
        args: RedisCommandArgs = ["FT.SEARCH", index_name, query]
        if (limit := config.get("limit")) is not None:
            args.extend(["LIMIT", str(config.get("offset", 0)), str(limit)])
        result = await redis.execute_command(*args)
        documents: list[dict[str, Any]] = []
        for index in range(1, len(result), 2):
            fields = result[index + 1]
            documents.append(
                {"id": result[index], **dict(zip(fields[::2], fields[1::2], strict=False))}
            )
        return {"totalResults": result[0], "documents": documents}

    return await _run("search_index", action)


async def delete_search_index(redis: Redis, index_name: str) -> CacheResult[dict[str, Any]]:
    async def action() -> dict[str, Any]:
        await redis.execute_command("FT.DROPINDEX", index_name)
        return {"deleted": True, "indexName": index_name}

    return await _run("delete_search_index", action)


async def create_bloom_filter(
    redis: Redis,
    filter_name: str,
    error_rate: float,
    capacity: int,
) -> CacheResult[dict[str, Any]]:
    async def action() -> dict[str, Any]:
        exists = await _bloom_filter_exists(redis, filter_name)
        if not exists:
            await redis.execute_command("BF.RESERVE", filter_name, str(error_rate), str(capacity))
        return {"created": not exists, "filterName": filter_name}

    return await _run("create_bloom_filter", action)


async def add_to_bloom_filter(
    redis: Redis,
    filter_name: str,
    value: Any | list[Any],
) -> CacheResult[bool | list[bool]]:
    async def action() -> bool | list[bool]:
        if isinstance(value, list):
            result = await redis.execute_command("BF.MADD", filter_name, *value)
            return [item == 1 for item in result]
        result = await redis.execute_command("BF.ADD", filter_name, value)
        return result == 1

    return await _run("add_to_bloom_filter", action)


async def check_bloom_filter(
    redis: Redis,
    filter_name: str,
    value: Any | list[Any],
) -> CacheResult[bool | list[bool]]:
    async def action() -> bool | list[bool]:
        if isinstance(value, list):
            result = await redis.execute_command("BF.MEXISTS", filter_name, *value)
            return [item == 1 for item in result]
        result = await redis.execute_command("BF.EXISTS", filter_name, value)
        return result == 1

    return await _run("check_bloom_filter", action)


async def get_bloom_filter_info(redis: Redis, filter_name: str) -> CacheResult[dict[str, Any]]:
    async def action() -> dict[str, Any]:
        result = await redis.execute_command("BF.INFO", filter_name)
        return {result[index]: result[index + 1] for index in range(0, len(result), 2)}

    return await _run("get_bloom_filter_info", action)


# Compatibility names for callers migrating from the intermediate Result facade.
set_cache_result = set_cache
get_cache_result = get_cache
delete_cache_result = delete_cache
set_hash_result = set_hash
get_hash_result = get_hash
update_hash_result = update_hash
push_to_list_result = push_to_list
get_list_items_result = get_list_items
search_index_result = search_index
add_to_bloom_filter_result = add_to_bloom_filter

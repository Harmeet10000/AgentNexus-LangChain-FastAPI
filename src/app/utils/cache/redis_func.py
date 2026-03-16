"""Redis cache utility functions.

This module provides reusable Redis operations with support for:
- String caching (SET, GET)
- Hash operations (HSET, HGET)
- List operations (LPUSH, RPUSH, LRANGE)
- Search indexes (FT.CREATE, FT.SEARCH)
- Bloom filters (BF.ADD, BF.EXISTS)

Use Case | Recommended Redis Type:
- Caching JWT tokens or API responses | String
- Caching user profiles (id, name, email) | Hash
- Caching recent notifications | List
- Caching online user IDs | Set
- Caching leaderboard scores | Sorted Set
"""

from typing import Any

import orjson
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.utils import logger
from app.utils.exceptions import DatabaseException

type CacheKeyPart = str | int
type CacheKey = CacheKeyPart | list[CacheKeyPart]
type RedisCommandArgs = list[str]

# ────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────


def _build_database_exception(detail: str, exc: Exception) -> DatabaseException:
    """Create a normalized database exception with the original error attached."""

    return DatabaseException(detail=detail, original_exc=exc)


def _redis_error_contains(exc: RedisError, message: str) -> bool:
    """Case-insensitive Redis error matching for command capability checks."""

    return message.lower() in str(exc).lower()




def get_key_name(object_type: str, *args: CacheKeyPart) -> str:
    """Generate a namespaced cache key from object type and args.

    Args:
        object_type: Entity type (e.g., 'user', 'token')
        *args: Arguments to append to key

    Returns:
        Formatted key like 'user:123:profile'
    """
    args_str = ":".join(str(arg) for arg in args)
    return f"{object_type}:{args_str}"


def get_cache_key(object_type: str, key: CacheKey) -> str:
    """Get a cache key handling single or multiple key parts.

    Args:
        object_type: Entity type
        key: Single key, ID, or list of key components

    Returns:
        Formatted cache key
    """
    if isinstance(key, list):
        return get_key_name(object_type, *key)
    return get_key_name(object_type, key)


def serialize_data(value: Any) -> str:
    """Serialize data to JSON string using orjson.

    Args:
        value: Data to serialize

    Returns:
        JSON string

    Raises:
        DatabaseException: If serialization fails
    """
    try:
        if isinstance(value, str):
            return value
        # orjson returns bytes, decode to string
        return orjson.dumps(value).decode("utf-8")
    except Exception as exc:
        raise _build_database_exception(
            detail=f"Failed to serialize data: {exc!s}",
            exc=exc,
        ) from exc


def deserialize_data(value: str | bytes) -> Any:
    """Deserialize JSON string using orjson.

    Args:
        value: JSON string or bytes

    Returns:
        Deserialized Python object

    Raises:
        DatabaseException: If deserialization fails
    """
    try:
        if isinstance(value, bytes):
            return orjson.loads(value)
        return orjson.loads(value.encode("utf-8") if isinstance(value, str) else value)
    except Exception as exc:
        raise _build_database_exception(
            detail=f"Failed to deserialize data: {exc!s}",
            exc=exc,
        ) from exc


# ────────────────────────────────────────────────────────────
# String Operations (SET, GET, DEL)
# ────────────────────────────────────────────────────────────


async def set_cache(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any,
    expire_seconds: int = 1800,
) -> bool:
    """Cache a value with expiration.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Cache key identifier(s)
        value: Value to cache (auto-serialized if object)
        expire_seconds: TTL in seconds (default: 30 min)

    Returns:
        True if cache was set successfully

    Raises:
        DatabaseException: If cache operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        string_value = serialize_data(value)
        await redis.set(cache_key, string_value, ex=expire_seconds)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to set cache: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to set cache for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(f"Cache set: {cache_key}", object_type=object_type, ttl=expire_seconds)
        return True


async def get_cache(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    parse_json: bool = True,
) -> Any:
    """Retrieve a cached value.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Cache key identifier(s)
        parse_json: Whether to deserialize JSON (default: True)

    Returns:
        Cached value or None if not found

    Raises:
        DatabaseException: If cache operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.get(cache_key)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to get cache: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to retrieve cache for {object_type}",
            exc=exc,
        ) from exc
    else:
        if result is None:
            return None

        if parse_json:
            return deserialize_data(result)

        return result


async def delete_cache(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> bool:
    """Delete a cached value.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Cache key identifier(s)

    Returns:
        True if at least one key was deleted

    Raises:
        DatabaseException: If cache operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.delete(cache_key)
    except Exception as exc:
        logger.error(
            f"Failed to delete cache: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to delete cache for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(f"Cache deleted: {cache_key}", deleted_count=result)
        return result > 0


# ────────────────────────────────────────────────────────────
# Pipeline Operations
# ────────────────────────────────────────────────────────────


async def execute_pipeline(
    redis: Redis,
    operations: list[dict[str, Any]],
) -> list[Any]:
    """Execute multiple Redis commands in a pipeline.

    Args:
        redis: Redis client instance
        operations: List of operations like:
            [
                {"command": "set", "args": ["key1", "value1"]},
                {"command": "hset", "args": ["hash1", "field", "value"]},
                {"command": "expire", "args": ["key1", 3600]}
            ]

    Returns:
        List of results from each operation

    Raises:
        DatabaseException: If pipeline execution fails
    """
    try:
        pipeline = redis.pipeline()

        for op in operations:
            command = op["command"]
            args = op["args"]
            getattr(pipeline, command)(*args)

        results = await pipeline.execute()
    except Exception as exc:
        logger.error(
            f"Pipeline execution failed with {len(operations)} operations",
            error=str(exc),
        )
        raise _build_database_exception(
            detail="Failed to execute Redis pipeline",
            exc=exc,
        ) from exc
    else:
        logger.info(f"Pipeline executed: {len(operations)} operations")
        return results


# ────────────────────────────────────────────────────────────
# Hash Operations (HSET, HGET, HUPDATE, HDEL)
# ────────────────────────────────────────────────────────────


async def set_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    data: dict[str, Any],
    expire_seconds: int = 1800,
) -> bool:
    """Store a hash (object-like) in Redis.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Hash key identifier(s)
        data: Dictionary to store
        expire_seconds: TTL in seconds (default: 30 min)

    Returns:
        True if hash was set

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        # Serialize nested objects to prevent data loss
        serialized_data = {k: serialize_data(v) for k, v in data.items()}

        await redis.hset(cache_key, mapping=serialized_data)  # type: ignore[union-attr]

        if expire_seconds:
            await redis.expire(cache_key, expire_seconds)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to set hash: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to set hash for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Hash set: {cache_key}",
            object_type=object_type,
            fields=len(data),
            ttl=expire_seconds,
        )
        return True


async def get_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> dict[str, Any] | None:
    """Retrieve a hash from Redis.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Hash key identifier(s)

    Returns:
        Dictionary with deserialized values, or None if not found

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.hgetall(cache_key)  # type: ignore[union-attr]
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to get hash: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to retrieve hash for {object_type}",
            exc=exc,
        ) from exc
    else:
        if not result:
            return None

        deserialized = {k: deserialize_data(v) for k, v in result.items()}
        logger.info(
            f"Hash retrieved: {cache_key}",
            object_type=object_type,
            fields=len(deserialized),
        )
        return deserialized


async def update_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    data: dict[str, Any],
) -> bool:
    """Update specific fields in a hash.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Hash key identifier(s)
        data: Fields to update with new values

    Returns:
        True if hash was updated

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        serialized_data = {k: serialize_data(v) for k, v in data.items()}
        await redis.hset(cache_key, mapping=serialized_data)  # type: ignore[union-attr]
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to update hash: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to update hash for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Hash updated: {cache_key}",
            object_type=object_type,
            fields_updated=len(data),
        )
        return True


async def delete_hash_field(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    field: str,
) -> bool:
    """Delete a specific field from a hash.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Hash key identifier(s)
        field: Field name to delete

    Returns:
        True if field was deleted

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.hdel(cache_key, field)  # type: ignore[union-attr]
    except Exception as exc:
        logger.error(
            f"Failed to delete hash field: {cache_key}",
            field=field,
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to delete hash field for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Hash field deleted: {cache_key}",
            field=field,
            deleted_count=result,
        )
        return result > 0


async def delete_hash(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> bool:
    """Delete an entire hash.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: Hash key identifier(s)

    Returns:
        True if hash was deleted

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.delete(cache_key)
    except Exception as exc:
        logger.error(
            f"Failed to delete hash: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to delete hash for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(f"Hash deleted: {cache_key}", deleted_count=result)
        return result > 0


# ────────────────────────────────────────────────────────────
# List Operations (LPUSH, RPUSH, LRANGE, LREM, LSET, LTRIM)
# ────────────────────────────────────────────────────────────


async def push_to_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any | list[Any],
    prepend: bool = False,
    expire_seconds: int | None = None,
) -> int:
    """Push value(s) to a Redis list.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)
        value: Single value or list of values to push
        prepend: If True, use LPUSH (add to start); if False, use RPUSH
        expire_seconds: Optional TTL in seconds

    Returns:
        New length of the list

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        # Handle multiple values
        values = value if isinstance(value, list) else [value]

        # Serialize objects, keep primitives as-is
        string_values = [serialize_data(v) for v in values]

        # LPUSH for prepending, RPUSH for appending
        if prepend:
            result = await redis.lpush(cache_key, *string_values)  # type: ignore[union-attr]
        else:
            result = await redis.rpush(cache_key, *string_values)  # type: ignore[union-attr]

        if expire_seconds:
            await redis.expire(cache_key, expire_seconds)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to push to list: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to push to list for {object_type}",
            exc=exc,
        ) from exc
    else:
        direction = "Prepended" if prepend else "Appended"
        logger.info(
            f"{direction} to list: {cache_key}",
            object_type=object_type,
            items_added=len(string_values),
            list_length=result,
        )
        return result


async def get_list_items(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    start: int = 0,
    end: int = -1,
    parse_json: bool = True,
) -> list[Any]:
    """Retrieve items from a Redis list.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)
        start: Starting index (default: 0)
        end: Ending index (default: -1 = all)
        parse_json: Whether to deserialize items (default: True)

    Returns:
        List of items (deserialized if parse_json=True)

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        items = await redis.lrange(cache_key, start, end)  # type: ignore[union-attr]
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to get list items: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to retrieve list items for {object_type}",
            exc=exc,
        ) from exc
    else:
        if not items:
            return []

        if parse_json:
            result = []
            for idx, item in enumerate(items):
                try:
                    result.append(deserialize_data(item))
                except Exception as exc:
                    logger.warning(
                        "Failed to deserialize list item",
                        cache_key=cache_key,
                        index=idx,
                        error=str(exc),
                    )
                    result.append(item)
            return result

        return items


async def get_list_length(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> int:
    """Get the length of a Redis list.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)

    Returns:
        List length

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        length = await redis.llen(cache_key)  # type: ignore[union-attr]
    except Exception as exc:
        logger.error(
            f"Failed to get list length: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to get list length for {object_type}",
            exc=exc,
        ) from exc
    else:
        return length


async def remove_from_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    value: Any,
    count: int = 0,
) -> int:
    """Remove items from a Redis list.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)
        value: Value to remove
        count: Number of occurrences to remove
              0 = all occurrences
              > 0 = first 'count' from head
              < 0 = last 'count' from tail

    Returns:
        Number of items removed

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        string_value = serialize_data(value)
        result = await redis.lrem(cache_key, count, string_value)  # type: ignore[union-attr]
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to remove from list: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to remove from list for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Removed from list: {cache_key}",
            object_type=object_type,
            items_removed=result,
        )
        return result


async def update_list_item(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    index: int,
    new_value: Any,
) -> bool:
    """Update a specific item in a Redis list by index.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)
        index: Position of item to update
        new_value: New value for the item

    Returns:
        True if update was successful

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        string_value = serialize_data(new_value)
        await redis.lset(cache_key, index, string_value)  # type: ignore[union-attr]
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to update list item: {cache_key}",
            index=index,
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to update list item for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Updated list item: {cache_key}",
            object_type=object_type,
            index=index,
        )
        return True


async def trim_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
    start: int,
    end: int,
) -> bool:
    """Trim a Redis list to a specified range.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)
        start: Starting index
        end: Ending index

    Returns:
        True if trim was successful

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        await redis.ltrim(cache_key, start, end)  # type: ignore[union-attr]
    except Exception as exc:
        logger.error(
            f"Failed to trim list: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to trim list for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Trimmed list: {cache_key}",
            object_type=object_type,
            range=f"{start}:{end}",
        )
        return True


async def delete_list(
    redis: Redis,
    object_type: str,
    key: CacheKey,
) -> bool:
    """Delete an entire Redis list.

    Args:
        redis: Redis client instance
        object_type: Entity type for namespacing
        key: List key identifier(s)

    Returns:
        True if list was deleted

    Raises:
        DatabaseException: If operation fails
    """
    cache_key = get_cache_key(object_type, key)

    try:
        result = await redis.delete(cache_key)
    except Exception as exc:
        logger.error(
            f"Failed to delete list: {cache_key}",
            error=str(exc),
            object_type=object_type,
        )
        raise _build_database_exception(
            detail=f"Failed to delete list for {object_type}",
            exc=exc,
        ) from exc
    else:
        logger.info(f"List deleted: {cache_key}", deleted_count=result)
        return result > 0


# ────────────────────────────────────────────────────────────
# Search Index Operations (RediSearch FT.*)
# ────────────────────────────────────────────────────────────
async def _search_index_exists(redis: Redis, index_name: str) -> bool:
    """Return whether a RediSearch index already exists."""

    try:
        await redis.execute_command("FT.INFO", index_name)
    except RedisError as exc:
        if _redis_error_contains(exc, "Unknown index name"):
            return False
        raise
    else:
        return True


async def _bloom_filter_exists(redis: Redis, filter_name: str) -> bool:
    """Return whether a Redis Bloom filter already exists."""

    try:
        await redis.execute_command("BF.INFO", filter_name)
    except RedisError as exc:
        if _redis_error_contains(exc, "not found"):
            return False
        raise
    else:
        return True


def _build_create_search_index_args(
    index_name: str,
    prefix: str | None,
    schema: dict[str, dict[str, Any]],
    options: dict[str, Any],
) -> RedisCommandArgs:
    """Build the FT.CREATE command arguments."""

    args: RedisCommandArgs = ["FT.CREATE", index_name]

    if prefix:
        args.extend(["ON", "HASH", "PREFIX", "1", prefix])

    language = options.get("language")
    if language:
        args.extend(["LANGUAGE", str(language)])

    stopwords = options.get("stopwords")
    if isinstance(stopwords, list) and stopwords:
        args.extend(["STOPWORDS", str(len(stopwords)), *[str(word) for word in stopwords]])

    args.append("SCHEMA")
    for field, definition in schema.items():
        args.extend([field, str(definition.get("type", "TEXT"))])

        if definition.get("sortable"):
            args.append("SORTABLE")
        if definition.get("noindex"):
            args.append("NOINDEX")
        if definition.get("nostem"):
            args.append("NOSTEM")

        weight = definition.get("weight")
        if weight is not None:
            args.extend(["WEIGHT", str(weight)])

        separator = definition.get("separator")
        if separator:
            args.extend(["SEPARATOR", str(separator)])

    return args


def _append_search_highlight_args(args: RedisCommandArgs, options: dict[str, Any]) -> None:
    """Append FT.SEARCH highlight options."""

    if not options.get("highlight"):
        return

    args.append("HIGHLIGHT")

    highlight_fields = options.get("highlightFields")
    if isinstance(highlight_fields, list) and highlight_fields:
        args.extend(
            ["FIELDS", str(len(highlight_fields)), *[str(field) for field in highlight_fields]]
        )

    highlight_tags = options.get("highlightTags")
    if isinstance(highlight_tags, list | tuple) and len(highlight_tags) == 2:
        args.extend(["TAGS", str(highlight_tags[0]), str(highlight_tags[1])])


def _append_search_summarize_args(args: RedisCommandArgs, options: dict[str, Any]) -> None:
    """Append FT.SEARCH summarize options."""

    if not options.get("summarize"):
        return

    args.append("SUMMARIZE")

    summarize_fields = options.get("summarizeFields")
    if isinstance(summarize_fields, list) and summarize_fields:
        args.extend(
            ["FIELDS", str(len(summarize_fields)), *[str(field) for field in summarize_fields]]
        )

    summarize_frags = options.get("summarizeFrags")
    if summarize_frags:
        args.extend(["FRAGS", str(summarize_frags)])

    summarize_len = options.get("summarizeLen")
    if summarize_len:
        args.extend(["LEN", str(summarize_len)])

    summarize_separator = options.get("summarizeSeparator")
    if summarize_separator:
        args.extend(["SEPARATOR", str(summarize_separator)])


def _append_search_filter_args(args: RedisCommandArgs, options: dict[str, Any]) -> None:
    """Append FT.SEARCH numeric and geo filters."""

    filters = options.get("filters")
    if isinstance(filters, list):
        for filter_config in filters:
            args.extend(
                [
                    "FILTER",
                    str(filter_config["field"]),
                    str(filter_config["min"]),
                    str(filter_config["max"]),
                ]
            )

    geo_filter = options.get("geoFilter")
    if isinstance(geo_filter, dict):
        args.extend(
            [
                "GEOFILTER",
                str(geo_filter["field"]),
                str(geo_filter.get("lon", 0)),
                str(geo_filter.get("lat", 0)),
                str(geo_filter.get("radius", 10)),
                str(geo_filter.get("unit", "m")),
            ]
        )


def _build_search_index_query_args(
    index_name: str,
    query: str,
    options: dict[str, Any],
) -> RedisCommandArgs:
    """Build the FT.SEARCH command arguments."""

    args: RedisCommandArgs = ["FT.SEARCH", index_name, query]

    limit = options.get("limit")
    if limit is not None:
        args.extend(["LIMIT", str(options.get("offset", 0)), str(limit)])

    sort_by = options.get("sortBy")
    if sort_by:
        args.extend(["SORTBY", str(sort_by)])
        sort_direction = options.get("sortDirection")
        if sort_direction:
            args.append(str(sort_direction).upper())

    return_fields = options.get("returnFields")
    if isinstance(return_fields, list) and return_fields:
        args.extend(["RETURN", str(len(return_fields)), *[str(field) for field in return_fields]])

    _append_search_highlight_args(args, options)
    _append_search_summarize_args(args, options)
    _append_search_filter_args(args, options)
    return args


def _parse_search_result(result: list[Any]) -> dict[str, Any]:
    """Convert the FT.SEARCH list response into a structured payload."""

    total_results = result[0]
    documents: list[dict[str, Any]] = []

    for index in range(1, len(result), 2):
        document_id = result[index]
        fields = result[index + 1]
        document = {"id": document_id}

        for field_index in range(0, len(fields), 2):
            document[fields[field_index]] = fields[field_index + 1]

        documents.append(document)

    return {"totalResults": total_results, "documents": documents}


async def create_search_index(
    redis: Redis,
    index_name: str,
    prefix: str | None,
    schema: dict[str, dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Redis search index (FT.CREATE).

    Args:
        redis: Redis client instance
        index_name: Name of the index
        prefix: Key prefix for indexed documents (e.g., "user:")
        schema: Field definitions dict like:
            {
                "email": {"type": "TEXT", "sortable": True},
                "age": {"type": "NUMERIC"},
                "tags": {"type": "TAG", "separator": ","}
            }
        options: Additional options like language, stopwords, etc.

    Returns:
        {"created": bool, "indexName": str}

    Raises:
        DatabaseException: If operation fails
    """
    index_exists = False

    try:
        normalized_options = options or {}
        index_exists = await _search_index_exists(redis, index_name)
        if not index_exists:
            args = _build_create_search_index_args(
                index_name=index_name,
                prefix=prefix,
                schema=schema,
                options=normalized_options,
            )
            await redis.execute_command(*args)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to create search index: {index_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to create search index {index_name}",
            exc=exc,
        ) from exc
    else:
        if index_exists:
            logger.info(f"Search index already exists: {index_name}")
            return {"created": False, "indexName": index_name}

        logger.info(
            f"Search index created: {index_name}",
            prefix=prefix,
            fields=len(schema),
        )
        return {"created": True, "indexName": index_name}


async def search_index(
    redis: Redis,
    index_name: str,
    query: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search a Redis search index (FT.SEARCH).

    Args:
        redis: Redis client instance
        index_name: Name of the index to search
        query: Search query string
        options: Search options like:
            {
                "limit": 10,
                "offset": 0,
                "sortBy": "updated_at",
                "sortDirection": "desc",
                "returnFields": ["id", "name"],
                "highlight": True,
                "highlightFields": ["title"],
                "filters": [{"field": "age", "min": 18, "max": 65}],
                "geoFilter": {"field": "location", "lon": 0, "lat": 0, "radius": 10, "unit": "km"}
            }

    Returns:
        {"totalResults": int, "documents": list[dict]}

    Raises:
        DatabaseException: If search fails
    """
    try:
        normalized_options = options or {}
        args = _build_search_index_query_args(index_name, query, normalized_options)
        result = await redis.execute_command(*args)
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to search index: {index_name}",
            query=query,
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to search index {index_name}",
            exc=exc,
        ) from exc
    else:
        parsed_result = _parse_search_result(result)
        logger.info(
            f"Search completed: {index_name}",
            query=query,
            total_results=parsed_result["totalResults"],
            returned=len(parsed_result["documents"]),
        )
        return parsed_result


async def delete_search_index(
    redis: Redis,
    index_name: str,
) -> dict[str, Any]:
    """Delete a Redis search index (FT.DROPINDEX).

    Args:
        redis: Redis client instance
        index_name: Name of the index to delete

    Returns:
        {"deleted": True, "indexName": str}

    Raises:
        DatabaseException: If operation fails
    """
    try:
        await redis.execute_command("FT.DROPINDEX", index_name)
    except Exception as exc:
        logger.error(
            f"Failed to delete search index: {index_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to delete search index {index_name}",
            exc=exc,
        ) from exc
    else:
        logger.info(f"Search index deleted: {index_name}")
        return {"deleted": True, "indexName": index_name}


# ────────────────────────────────────────────────────────────
# Bloom Filter Operations (BF.*)
# ────────────────────────────────────────────────────────────


async def create_bloom_filter(
    redis: Redis,
    filter_name: str,
    error_rate: float,
    capacity: int,
) -> dict[str, Any]:
    """Create a Bloom filter (BF.RESERVE).

    Args:
        redis: Redis client instance
        filter_name: Name of the filter
        error_rate: Error rate (e.g., 0.01 for 1%)
        capacity: Expected capacity

    Returns:
        {"created": bool, "filterName": str}

    Raises:
        DatabaseException: If operation fails
    """
    filter_exists = False

    try:
        filter_exists = await _bloom_filter_exists(redis, filter_name)
        if not filter_exists:
            await redis.execute_command(
                "BF.RESERVE",
                filter_name,
                str(error_rate),
                str(capacity),
            )
    except DatabaseException:
        raise
    except Exception as exc:
        logger.error(
            f"Failed to create bloom filter: {filter_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to create bloom filter {filter_name}",
            exc=exc,
        ) from exc
    else:
        if filter_exists:
            logger.info(f"Bloom filter already exists: {filter_name}")
            return {"created": False, "filterName": filter_name}

        logger.info(
            f"Bloom filter created: {filter_name}",
            error_rate=error_rate,
            capacity=capacity,
        )
        return {"created": True, "filterName": filter_name}


async def add_to_bloom_filter(
    redis: Redis,
    filter_name: str,
    value: Any | list[Any],
) -> bool | list[bool]:
    """Add item(s) to a Bloom filter (BF.ADD or BF.MADD).

    Args:
        redis: Redis client instance
        filter_name: Name of the filter
        value: Single value or list of values to add

    Returns:
        True/False for single value, list of bools for multiple values

    Raises:
        DatabaseException: If operation fails
    """
    result_value: bool | list[bool]

    try:
        if isinstance(value, list):
            result = await redis.execute_command("BF.MADD", filter_name, *value)
            logger.info(
                f"Added multiple items to bloom filter: {filter_name}",
                items_added=sum(1 for r in result if r == 1),
            )
            result_value = [r == 1 for r in result]
        else:
            result = await redis.execute_command("BF.ADD", filter_name, value)
            result_value = result == 1
            logger.info(
                f"Added to bloom filter: {filter_name}",
                is_new=result_value,
            )
    except Exception as exc:
        logger.error(
            f"Failed to add to bloom filter: {filter_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to add to bloom filter {filter_name}",
            exc=exc,
        ) from exc
    else:
        return result_value


async def check_bloom_filter(
    redis: Redis,
    filter_name: str,
    value: Any | list[Any],
) -> bool | list[bool]:
    """Check if item(s) might exist in a Bloom filter (BF.EXISTS or BF.MEXISTS).

    Args:
        redis: Redis client instance
        filter_name: Name of the filter
        value: Single value or list of values to check

    Returns:
        True/False for single value, list of bools for multiple values

    Raises:
        DatabaseException: If operation fails
    """
    result_value: bool | list[bool]

    try:
        if isinstance(value, list):
            result = await redis.execute_command("BF.MEXISTS", filter_name, *value)
            logger.info(
                f"Checked multiple items in bloom filter: {filter_name}",
                exists_count=sum(1 for r in result if r == 1),
            )
            result_value = [r == 1 for r in result]
        else:
            result = await redis.execute_command("BF.EXISTS", filter_name, value)
            result_value = result == 1
            logger.info(
                f"Checked bloom filter: {filter_name}",
                exists=result_value,
            )
    except Exception as exc:
        logger.error(
            f"Failed to check bloom filter: {filter_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to check bloom filter {filter_name}",
            exc=exc,
        ) from exc
    else:
        return result_value


async def get_bloom_filter_info(
    redis: Redis,
    filter_name: str,
) -> dict[str, Any]:
    """Get Bloom filter info (BF.INFO).

    Args:
        redis: Redis client instance
        filter_name: Name of the filter

    Returns:
        Dictionary with filter metadata

    Raises:
        DatabaseException: If operation fails
    """
    try:
        result = await redis.execute_command("BF.INFO", filter_name)

        # Convert list response to dict: [key1, val1, key2, val2, ...]
        info = {}
        for i in range(0, len(result), 2):
            info[result[i]] = result[i + 1]
    except Exception as exc:
        logger.error(
            f"Failed to get bloom filter info: {filter_name}",
            error=str(exc),
        )
        raise _build_database_exception(
            detail=f"Failed to get bloom filter info for {filter_name}",
            exc=exc,
        ) from exc
    else:
        logger.info(
            f"Retrieved bloom filter info: {filter_name}",
            capacity=info.get("Capacity"),
        )
        return info

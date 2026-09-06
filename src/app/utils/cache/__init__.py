"""Cache utilities using Redis."""

from redis.asyncio import Redis

from app.utils.cache.errors import CacheCode, CacheError, CacheResult
from app.utils.cache.redis_func import (
    add_to_bloom_filter,
    add_to_bloom_filter_result,
    check_bloom_filter,
    # Bloom filter operations
    create_bloom_filter,
    # Search index operations
    create_search_index,
    delete_cache,
    delete_cache_result,
    delete_hash,
    delete_hash_field,
    delete_list,
    delete_search_index,
    deserialize_data,
    # Pipeline operations
    execute_pipeline,
    get_bloom_filter_info,
    get_cache,
    get_cache_key,
    get_cache_result,
    get_hash,
    get_hash_result,
    # Utility functions
    get_key_name,
    get_list_items,
    get_list_items_result,
    get_list_length,
    # List operations
    push_to_list,
    push_to_list_result,
    remove_from_list,
    search_index,
    search_index_result,
    serialize_data,
    # String operations
    set_cache,
    set_cache_result,
    # Hash operations
    set_hash,
    set_hash_result,
    trim_list,
    update_hash,
    update_hash_result,
    update_list_item,
)
from app.utils.cache.redis_guard_adapter import RedisGuardAdapter
from app.utils.cache.result import run_cache_operation

__all__ = [  # noqa: RUF022
    "Redis",
    "RedisGuardAdapter",
    "CacheCode",
    "CacheError",
    "CacheResult",
    "add_to_bloom_filter",
    "check_bloom_filter",
    # Bloom filter operations
    "create_bloom_filter",
    # Search index operations
    "create_search_index",
    "delete_cache",
    "delete_cache_result",
    "delete_hash",
    "delete_hash_field",
    "delete_list",
    "delete_search_index",
    "deserialize_data",
    # Pipeline operations
    "execute_pipeline",
    "get_bloom_filter_info",
    "get_cache",
    "get_cache_result",
    "get_cache_key",
    "get_hash",
    "get_hash_result",
    # Utility functions
    "get_key_name",
    "get_list_items",
    "get_list_items_result",
    "get_list_length",
    # List operations
    "push_to_list",
    "push_to_list_result",
    "remove_from_list",
    "search_index",
    "search_index_result",
    "serialize_data",
    # String operations
    "set_cache",
    "set_cache_result",
    # Hash operations
    "set_hash",
    "set_hash_result",
    "trim_list",
    "update_hash",
    "update_hash_result",
    "update_list_item",
    "run_cache_operation",
]

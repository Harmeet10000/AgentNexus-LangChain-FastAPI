"""Cache utilities using Redis."""

from app.utils.cache.redis_func import (
    add_to_bloom_filter,
    check_bloom_filter,
    # Bloom filter operations
    create_bloom_filter,
    # Search index operations
    create_search_index,
    delete_cache,
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
    get_hash,
    # Utility functions
    get_key_name,
    get_list_items,
    get_list_length,
    # List operations
    push_to_list,
    remove_from_list,
    search_index,
    serialize_data,
    # String operations
    set_cache,
    # Hash operations
    set_hash,
    trim_list,
    update_hash,
    update_list_item,
)
from app.utils.cache.redis_protocol_adapter import RedisProtocolAdapter

__all__ = [
    "RedisProtocolAdapter",
    "add_to_bloom_filter",
    "check_bloom_filter",
    # Bloom filter operations
    "create_bloom_filter",
    # Search index operations
    "create_search_index",
    "delete_cache",
    "delete_hash",
    "delete_hash_field",
    "delete_list",
    "delete_search_index",
    "deserialize_data",
    # Pipeline operations
    "execute_pipeline",
    "get_bloom_filter_info",
    "get_cache",
    "get_cache_key",
    "get_hash",
    # Utility functions
    "get_key_name",
    "get_list_items",
    "get_list_length",
    # List operations
    "push_to_list",
    "remove_from_list",
    "search_index",
    "serialize_data",
    # String operations
    "set_cache",
    # Hash operations
    "set_hash",
    "trim_list",
    "update_hash",
    "update_list_item",
]

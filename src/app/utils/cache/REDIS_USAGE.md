# Redis Cache Utilities - Usage Guide

This module provides comprehensive Redis operations for caching, searching, and filtering data in your FastAPI application.

## Access Redis Client

Redis is available from `app.state.redis` in your FastAPI application:

```python
from fastapi import Depends, Request
from redis.asyncio import Redis

def get_redis(request: Request) -> Redis:
    return request.app.state.redis
```

## String Caching (Recommended for simple values & JSON)

### Set Cache
```python
from app.utils.cache import set_cache

# Simple value (auto-serialized with orjson)
await set_cache(redis, "user", user_id, {"name": "John", "email": "john@example.com"})

# With custom TTL (default: 1800 seconds = 30 minutes)
await set_cache(redis, "session", session_id, session_data, expire_seconds=3600)

# Cache key becomes: "user:{user_id}"
```

### Get Cache
```python
from app.utils.cache import get_cache

# Retrieve and auto-deserialize
user_data = await get_cache(redis, "user", user_id)  # Returns dict or None

# Get without deserialization
raw_value = await get_cache(redis, "token", token_id, parse_json=False)
```

### Delete Cache
```python
from app.utils.cache import delete_cache

deleted = await delete_cache(redis, "user", user_id)  # Returns True if deleted
```

## Hash Operations (Recommended for objects with multiple fields)

Use for object-like structures where you need partial reads/updates without deserializing the entire object.

### Set Hash
```python
from app.utils.cache import set_hash

user_profile = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "123-456-7890",
    "age": 30,
    "tags": ["admin", "premium"]
}

await set_hash(redis, "user_profile", user_id, user_profile, expire_seconds=3600)
# Stores in Redis as HASH: "user_profile:{user_id}"
```

### Get Hash
```python
from app.utils.cache import get_hash

# Retrieve all fields and auto-deserialize
profile = await get_hash(redis, "user_profile", user_id)
# Returns: {"name": "John Doe", "email": "john@example.com", ...}
```

### Update Hash Fields
```python
from app.utils.cache import update_hash

# Update specific fields only (doesn't affect other fields)
await update_hash(redis, "user_profile", user_id, {
    "email": "newemail@example.com",
    "phone": "999-999-9999"
})
```

### Delete Hash Field
```python
from app.utils.cache import delete_hash_field

await delete_hash_field(redis, "user_profile", user_id, "phone")
```

### Delete Hash
```python
from app.utils.cache import delete_hash

await delete_hash(redis, "user_profile", user_id)
```

## List Operations (For ordered sequences)

Use for feeds, notifications, chat messages, activity logs, etc.

### Push Items
```python
from app.utils.cache import push_to_list

# Append to end (RPUSH)
await push_to_list(redis, "notifications", user_id, {
    "id": "notif_123",
    "message": "You have a new message",
    "timestamp": "2024-01-15T10:30:00Z"
})

# Multiple items at once
notifications = [
    {"id": "n1", "message": "Message 1"},
    {"id": "n2", "message": "Message 2"}
]
await push_to_list(redis, "notifications", user_id, notifications)

# Prepend to start (LPUSH)
await push_to_list(redis, "recent_activity", user_id, activity_item, prepend=True)

# With TTL
await push_to_list(redis, "session_history", session_id, item, expire_seconds=86400)
```

### Get List Items
```python
from app.utils.cache import get_list_items

# Get all items
all_notifications = await get_list_items(redis, "notifications", user_id)

# Get range (like pagination)
recent = await get_list_items(redis, "notifications", user_id, start=0, end=9)  # First 10

# Get without deserialization
raw_items = await get_list_items(redis, "notifications", user_id, parse_json=False)
```

### List Length
```python
from app.utils.cache import get_list_length

count = await get_list_length(redis, "notifications", user_id)
```

### Remove Items
```python
from app.utils.cache import remove_from_list

# Remove all occurrences of a value
await remove_from_list(redis, "notifications", user_id, notification_item)

# Remove first 5 occurrences from start
await remove_from_list(redis, "notifications", user_id, item, count=5)

# Remove last 3 occurrences from end
await remove_from_list(redis, "notifications", user_id, item, count=-3)
```

### Update List Item
```python
from app.utils.cache import update_list_item

# Update item at specific index
await update_list_item(redis, "notifications", user_id, 0, updated_notification)
```

### Trim List
```python
from app.utils.cache import trim_list

# Keep only first 100 items
await trim_list(redis, "notifications", user_id, 0, 99)

# Keep middle items (remove first and last)
await trim_list(redis, "notifications", user_id, 10, -10)
```

## Pipeline Operations (For multiple operations at once)

Execute multiple Redis commands in a single round-trip:

```python
from app.utils.cache import execute_pipeline

operations = [
    {"command": "set", "args": ["user:1", '"John"']},
    {"command": "set", "args": ["user:2", '"Jane"']},
    {"command": "expire", "args": ["user:1", "3600"]},
    {"command": "hset", "args": ["profile:1", "name", '"John"', "age", "30"]}
]

results = await execute_pipeline(redis, operations)
```

## Search Indexes (For full-text search on hashes)

Requires Redis 6.0+ with RediSearch module.

### Create Index
```python
from app.utils.cache import create_search_index

index = await create_search_index(
    redis,
    index_name="users_idx",
    prefix="user:",  # Index all keys starting with "user:"
    schema={
        "email": {"type": "TEXT", "sortable": True},
        "name": {"type": "TEXT"},
        "age": {"type": "NUMERIC", "sortable": True},
        "tags": {"type": "TAG", "separator": ","},
        "created_at": {"type": "NUMERIC"}
    },
    options={"language": "english"}
)
```

### Search Index
```python
from app.utils.cache import search_index

results = await search_index(
    redis,
    index_name="users_idx",
    query="john",  # Full-text search
    options={
        "limit": 10,
        "offset": 0,
        "sortBy": "age",
        "sortDirection": "asc",
        "returnFields": ["name", "email", "age"],
        "highlight": True,
        "highlightFields": ["name"],
        "filters": [
            {"field": "age", "min": 18, "max": 65}
        ]
    }
)

# Returns: {"totalResults": 42, "documents": [...]}
```

### Delete Index
```python
from app.utils.cache import delete_search_index

await delete_search_index(redis, "users_idx")
```

## Bloom Filters (For membership testing)

Probabilistic data structure for testing if an element might be in a set (no false negatives, some false positives).

### Create Bloom Filter
```python
from app.utils.cache import create_bloom_filter

# Error rate = 1%, capacity = 1,000,000 items
filter_info = await create_bloom_filter(
    redis,
    filter_name="active_users",
    error_rate=0.01,
    capacity=1000000
)
```

### Add Items
```python
from app.utils.cache import add_to_bloom_filter

# Single item
await add_to_bloom_filter(redis, "active_users", user_id)

# Multiple items
user_ids = [1, 2, 3, 4, 5]
results = await add_to_bloom_filter(redis, "active_users", user_ids)
# Returns: [True, True, False, ...]  # True if newly added, False if already existed
```

### Check Items
```python
from app.utils.cache import check_bloom_filter

# Single check
exists = await check_bloom_filter(redis, "active_users", user_id)  # True/False

# Multiple checks
user_ids = [1, 2, 3]
results = await check_bloom_filter(redis, "active_users", user_ids)  # [True, False, True]
```

### Get Filter Info
```python
from app.utils.cache import get_bloom_filter_info

info = await get_bloom_filter_info(redis, "active_users")
# {"Capacity": 1000000, "Size": 12500, "Number of items inserted": 50000, ...}
```

## Real-World Examples

### Caching User Sessions
```python
from fastapi import APIRouter, Depends
from app.utils.cache import set_cache, get_cache, delete_cache

router = APIRouter()

@router.post("/login")
async def login(credentials: LoginRequest, redis = Depends(get_redis)):
    # Validate credentials (DB lookup)
    user = await validate_user(credentials)
    
    # Create session
    session = {
        "user_id": user.id,
        "email": user.email,
        "login_time": datetime.utcnow().isoformat(),
        "ip": request.client.host
    }
    
    # Cache with 1 day TTL
    await set_cache(redis, "session", session_id, session, expire_seconds=86400)
    
    return {"session_id": session_id, "user": user}

@router.post("/logout")
async def logout(session_id: str, redis = Depends(get_redis)):
    await delete_cache(redis, "session", session_id)
    return {"status": "logged_out"}
```

### Recent Activity Feed
```python
from app.utils.cache import push_to_list, get_list_items, trim_list

async def add_activity(user_id: str, activity: dict, redis = Depends(get_redis)):
    # Add to feed (most recent first)
    await push_to_list(
        redis,
        "activity_feed",
        user_id,
        activity,
        prepend=True,
        expire_seconds=86400
    )
    
    # Keep only last 500 items
    await trim_list(redis, "activity_feed", user_id, 0, 499)

async def get_feed(user_id: str, redis = Depends(get_redis)):
    # Get first 20 items
    return await get_list_items(redis, "activity_feed", user_id, start=0, end=19)
```

### User Profile with Selective Updates
```python
from app.utils.cache import set_hash, get_hash, update_hash

async def cache_user_profile(user_id: str, user_data: dict, redis = Depends(get_redis)):
    # Cache entire profile
    await set_hash(
        redis,
        "profile",
        user_id,
        user_data,
        expire_seconds=3600
    )

async def update_user_email(user_id: str, new_email: str, redis = Depends(get_redis)):
    # Update only email field without affecting others
    await update_hash(redis, "profile", user_id, {"email": new_email})

async def get_user_profile(user_id: str, redis = Depends(get_redis)):
    profile = await get_hash(redis, "profile", user_id)
    if profile:
        return profile
    
    # Cache miss - fetch from DB and cache
    profile = await db.get_user(user_id)
    await cache_user_profile(user_id, profile)
    return profile
```

## Key Naming Convention

All functions use a consistent naming pattern:

```
{object_type}:{key_component}:{optional_sub_keys}

Examples:
- user:123:profile
- user:123
- session:abc-def-ghi
- notifications:user_456
- activity_feed:user_789:2024
```

## Error Handling

All functions raise `DatabaseException` on failure:

```python
from app.utils.exceptions import DatabaseException

try:
    data = await get_cache(redis, "user", user_id)
except DatabaseException as e:
    logger.error(f"Cache operation failed: {e.detail}")
    # Fallback to database
    data = await db.get_user(user_id)
```

## Best Practices

1. **Choose the Right Data Type**:
   - String: Simple values, JSON objects
   - Hash: Object-like data with many fields
   - List: Ordered sequences (feeds, logs)
   - Set: Unique items (tags, online users)
   - Bloom Filter: Membership testing

2. **TTL Management**: Always set appropriate TTLs to prevent memory bloat
   - Session data: 24-48 hours
   - User profiles: 1-6 hours
   - Activity feeds: 7-30 days
   - Temporary caches: 5-30 minutes

3. **Error Handling**: Always handle `DatabaseException` and have DB fallbacks

4. **Key Naming**: Use consistent patterns like `type:id:attribute`

5. **Serialization**: Uses `orjson` (fast JSON library) automatically

6. **Performance**: Use pipelines for multiple operations in one round-trip

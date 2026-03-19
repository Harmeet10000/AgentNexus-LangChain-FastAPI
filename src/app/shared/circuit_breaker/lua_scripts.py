# Returns: 1 (ALLOW), 2 (ALLOW_PROBE), 0 (REJECT)
CB_ACQUIRE_SCRIPT = """
local state_key = KEYS[1]
local timeout_key = KEYS[2]
local now_ms = tonumber(ARGV[1])
local probe_ttl_ms = tonumber(ARGV[2])

local state = redis.call("GET", state_key)

if not state or state == "CLOSED" then
    return 1
end

if state == "OPEN" then
    local timeout = tonumber(redis.call("GET", timeout_key) or 0)
    if now_ms >= timeout then
        -- Transition to HALF_OPEN and set a Mutex TTL for the probe
        redis.call("SET", state_key, "HALF_OPEN")
        redis.call("SET", timeout_key, now_ms + probe_ttl_ms)
        return 2
    else
        return 0
    end
end

if state == "HALF_OPEN" then
    local timeout = tonumber(redis.call("GET", timeout_key) or 0)
    if now_ms >= timeout then
        -- The previous probe worker died/timed out. Re-acquire the probe Mutex.
        redis.call("SET", timeout_key, now_ms + probe_ttl_ms)
        return 2
    end
    -- Mutex is currently held by another worker
    return 0
end
"""

CB_SUCCESS_SCRIPT = """
local state_key = KEYS[1]
local failures_key = KEYS[2]

redis.call("SET", state_key, "CLOSED")
redis.call("DEL", failures_key)
return 1
"""

CB_FAILURE_SCRIPT = """
local state_key = KEYS[1]
local failures_key = KEYS[2]
local timeout_key = KEYS[3]
local now_ms = tonumber(ARGV[1])
local failure_threshold = tonumber(ARGV[2])
local recovery_timeout_ms = tonumber(ARGV[3])

local state = redis.call("GET", state_key)

if state == "HALF_OPEN" then
    -- The probe failed. Immediately trip back to OPEN.
    redis.call("SET", state_key, "OPEN")
    redis.call("SET", timeout_key, now_ms + recovery_timeout_ms)
    return 1
end

if not state or state == "CLOSED" then
    local failures = redis.call("INCR", failures_key)
    if failures >= failure_threshold then
        -- Threshold reached. Trip the breaker.
        redis.call("SET", state_key, "OPEN")
        redis.call("SET", timeout_key, now_ms + recovery_timeout_ms)
    end
    return 1
end

return 1
"""

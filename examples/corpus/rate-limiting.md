# Rate limiting APIs

## Token bucket
A token bucket refills at a steady rate and allows short bursts up to its capacity, which
matches how real clients behave.

## Sliding window
A sliding window log counts requests in the last N seconds exactly, at the cost of storing a
timestamp per request.

## Distributed limits
When several API replicas share a limit, keep counters in a central store such as Redis and
update them atomically with a Lua script.

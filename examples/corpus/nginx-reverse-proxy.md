# nginx as a reverse proxy

## Proxy headers
When nginx forwards requests upstream it should pass `Host`, `X-Real-IP` and
`X-Forwarded-For` so the application sees the original client address and hostname.

## WebSocket upgrades
WebSockets need `proxy_http_version 1.1` plus the `Upgrade` and `Connection` headers,
otherwise the handshake is downgraded and the connection closes immediately.

## Timeouts and buffering
Long-running streaming responses, such as server-sent events or token streaming from a
language model, require `proxy_buffering off` and a generous `proxy_read_timeout`.

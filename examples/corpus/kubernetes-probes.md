# Kubernetes health probes

## Liveness versus readiness
A failing liveness probe restarts the container; a failing readiness probe only removes the
pod from Service endpoints. Use readiness for temporary unavailability such as warming caches.

## Startup probes
Slow-starting applications should use a startup probe so the liveness probe does not kill
them before initialisation finishes.

## Common mistakes
Never make liveness depend on external services like the database, otherwise one database
outage restarts every pod in the cluster and makes recovery slower.

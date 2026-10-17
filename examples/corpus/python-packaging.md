# Python packaging

## pyproject.toml
Modern projects declare metadata, dependencies and build backend in pyproject.toml.
Backends such as hatchling or setuptools build wheels from it.

## Virtual environments
Isolate each project's dependencies in a virtual environment so upgrading a library for one
project does not break another.

## Pinning dependencies
Applications should lock exact versions for reproducible deployments, while libraries should
declare compatible ranges so they do not conflict with other packages.

# Building small container images

## Multi-stage builds
Compile in a builder stage that has the full toolchain, then copy only the resulting
binary into a minimal runtime stage such as distroless or alpine. The final image no
longer contains compilers, package caches or source code.

## Layer caching
Order Dockerfile instructions from least to most frequently changing. Copy dependency
manifests and install dependencies before copying the application source so code edits
do not invalidate the dependency layer.

## Running as non-root
Add a dedicated user and switch to it with the USER instruction. Containers that run as
root make container escapes far more dangerous.

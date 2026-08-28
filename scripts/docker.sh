#!/usr/bin/env bash

# Docker build / run / debug cheat sheet for this project.
# Versioning: every image is tagged `sha-<short>` (immutable) + optional semver.
#   `make image-tag` prints the tag, `make image-build` / `image-build-prod` build it,
#   CI (docker.yml) pushes `sha-*`, `v*.*.*`, `latest` (main only) to GHCR.
# This file is intentionally a reference — copy the commands you need.

set -euo pipefail

# ------------------------------------------------------------------------------
# 0. Version helpers (single source of truth — matches Makefile)
# ------------------------------------------------------------------------------

# Short SHA + -dirty if working tree is dirty (what CI tags as sha-xxxx)
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git diff --quiet 2>/dev/null || echo -dirty)"
IMAGE_TAG="sha-${GIT_SHA}${DIRTY}"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
APP_VERSION="$(grep -Po '(?<=^version = ")[^"]*' pyproject.toml 2>/dev/null || echo 1.0.0)"
REGISTRY="ghcr.io/harmeet10000/langchain-fastapi-production"

echo "# QOL: computed tags for this checkout"
echo "#   IMAGE_TAG=$IMAGE_TAG  APP_VERSION=$APP_VERSION  BUILD_DATE=$BUILD_DATE"
echo "#   make image-tag  → $IMAGE_TAG  (canonical)"
echo ""

# ------------------------------------------------------------------------------
# 1. Project image lifecycle — dev (compose, preferred)
# ------------------------------------------------------------------------------

# All compose files now carry `image: ghcr.io/...:${IMAGE_TAG:-dev}` + build args
# so `docker inspect` and `curl /` answer the same question: "what commit am I?"
# ponytail: dev compose builds, prod compose pulls — never `latest` in prod.

# Build + run the full stack (dev, with bind-mounts) — git SHA baked as label + env
GIT_SHA="${GIT_SHA}${DIRTY}" BUILD_DATE="$BUILD_DATE" APP_VERSION="$APP_VERSION" IMAGE_TAG="$IMAGE_TAG" docker compose up --build -d

# Verify what you just built — without running the container
docker inspect "$REGISTRY:$IMAGE_TAG" --format '{{json .Config.Labels}}' | jq
#   expect: org.opencontainers.image.revision, version, created, source

# Verify at runtime — the app exposes the same values
curl -s http://localhost:5000/ | jq        # {version, git_sha, build_date}
curl -s http://localhost:5000/health | jq  # {version, gitSha, buildDate, status}

# Tag rollback (prod) — pull + run an immutable previous image
# IMAGE_TAG=sha-<previous> docker compose -f docker-compose.prod.yml up -d
# Never use :latest in prod compose.

# ------------------------------------------------------------------------------
# 2. Direct docker build (without compose —matches Makefile targets)
# ------------------------------------------------------------------------------

# Dev image (with hot-reload, src bind-mount at runtime via compose)
docker build \
  -f docker/dev.Dockerfile \
  --target dev \
  --build-arg GIT_SHA="${GIT_SHA}${DIRTY}" \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  --build-arg APP_VERSION="$APP_VERSION" \
  -t "$REGISTRY:$IMAGE_TAG" \
  -t "$REGISTRY:dev" \
  .

# Prod image (slim, non-root, no uv run at runtime)
docker build \
  -f docker/prod.Dockerfile \
  --target production \
  --build-arg GIT_SHA="${GIT_SHA}${DIRTY}" \
  --build-arg BUILD_DATE="$BUILD_DATE" \
  --build-arg APP_VERSION="$APP_VERSION" \
  -t "$REGISTRY:$IMAGE_TAG" \
  .

# Run prod image standalone (env-file must be .env.production or equivalent)
docker run -d \
  -p 5000:5000 \
  --name langchain-fastapi-server \
  --env-file .env.production \
  -e GIT_SHA="${GIT_SHA}${DIRTY}" \
  -e APP_VERSION="$APP_VERSION" \
  "$REGISTRY:$IMAGE_TAG"

# Follow logs
docker logs -f langchain-fastapi-server

# ------------------------------------------------------------------------------
# 3. Registry — GHCR (CI pushes; local push is manual)
# ------------------------------------------------------------------------------

# Login once (uses GITHUB_TOKEN in CI, PAT locally)
echo "$GITHUB_TOKEN" | docker login ghcr.io -u harmeet10000 --password-stdin

# Tag + push (usually CI does this — local only for hotfix)
docker tag "$REGISTRY:$IMAGE_TAG" "$REGISTRY:$IMAGE_TAG"
docker push "$REGISTRY:$IMAGE_TAG"

# For Docker Hub mirror (optional)
docker tag "$REGISTRY:$IMAGE_TAG" "harmeet10000/langchain-fastapi-server:$IMAGE_TAG"
docker push "harmeet10000/langchain-fastapi-server:$IMAGE_TAG"

# For AWS ECR (replace URI)
# docker tag "$REGISTRY:$IMAGE_TAG" "123456789012.dkr.ecr.us-east-1.amazonaws.com/langchain-fastapi-server:$IMAGE_TAG"
# docker push "123456789012.dkr.ecr.us-east-1.amazonaws.com/langchain-fastapi-server:$IMAGE_TAG"

# ------------------------------------------------------------------------------
# 4. Make shortcuts (canonical — keep this file in sync)
# ------------------------------------------------------------------------------

# make image-tag        # prints sha-xxxx[-dirty]
# make image-build      # dev image  → ghcr.io/...:sha-xxxx + :dev
# make image-build-prod # prod image → ghcr.io/...:sha-xxxx
# make image-push       # pushes ghcr.io/...:sha-xxxx (fails if sha-unknown)

# ------------------------------------------------------------------------------
# 5. Common variables for debugging (unchanged)
# ------------------------------------------------------------------------------

# Set the target container once and reuse it everywhere below.
CTR=langchain-fastapi-server

# Container ID is often easier to pass into lower-level commands.
CID="$(docker ps -qf "name=^${CTR}$")"

# PID of the container's init process on the host.
PID="$(docker inspect --format '{{.State.Pid}}' "$CTR" 2>/dev/null || echo 0)"

# Full inspect output as pretty JSON.
docker inspect "$CTR" | jq

# Interactive tree-style JSON inspection for very large inspect output.
docker inspect "$CTR" | jless

# If you have the `js` interactive JSON browser installed, this is also useful.
docker inspect "$CTR" | js

# ------------------------------------------------------------------------------
# 6. Inspecting container configuration without entering the container
# ------------------------------------------------------------------------------

# Show the entire Config object only.
docker inspect --format '{{json .Config}}' "$CTR" | jq

# Show environment variables (includes GIT_SHA, APP_VERSION, BUILD_DATE).
docker inspect --format '{{json .Config.Env}}' "$CTR" | jq -r '.[]' | grep -E 'GIT_SHA|APP_VERSION|BUILD_DATE'

# Show OCI labels (immutable build identity).
docker inspect --format '{{json .Config.Labels}}' "$CTR" | jq

# Show entrypoint and command exactly as Docker sees them.
docker inspect --format '{{json .Config.Entrypoint}}' "$CTR" | jq
docker inspect --format '{{json .Config.Cmd}}' "$CTR" | jq

# Show current status, restart count, and whether Docker thinks it is healthy.
docker inspect --format '{{json .State}}' "$CTR" | jq '{Status, Running, Restarting, OOMKilled, ExitCode, Error, StartedAt, FinishedAt, Health}'

# Show the container IP addresses on all attached networks.
docker inspect --format '{{json .NetworkSettings.Networks}}' "$CTR" | jq 'with_entries(.value |= {IPAddress, Gateway, MacAddress, Aliases})'

# Show published host ports and bindings.
docker inspect --format '{{json .NetworkSettings.Ports}}' "$CTR" | jq

# Show mounts, bind mounts, named volumes, and destinations.
docker inspect --format '{{json .Mounts}}' "$CTR" | jq

# Find the log file path Docker is writing on the host.
docker inspect --format '{{.LogPath}}' "$CTR"

# Show the exact image digest/container image being used.
docker inspect --format 'image={{.Config.Image}} id={{.Image}}' "$CTR"

# ------------------------------------------------------------------------------
# 7. Minimal / scratch container filesystem access from the host
# ------------------------------------------------------------------------------

# MergedDir gives you the live mounted root filesystem for overlay-based storage.
MERGED_DIR="$(docker inspect --format '{{.GraphDriver.Data.MergedDir}}' "$CTR" 2>/dev/null || echo /tmp)"
echo "$MERGED_DIR"

# List files in the container rootfs directly from the host.
sudo ls -lah "$MERGED_DIR"

# Inspect common paths even if the container has no shell.
sudo ls -lah "$MERGED_DIR/app"
sudo ls -lah "$MERGED_DIR/etc"

# Copy files out of the container without needing a shell in the image.
docker cp "$CTR":/app ./tmp-app-copy

# Export the full container filesystem as a tar archive for offline inspection.
docker export "$CTR" -o "/tmp/${CTR}.tar"

# Show what changed in the writable layer since the container started.
docker diff "$CTR"

# ------------------------------------------------------------------------------
# 8. Network troubleshooting for containers that have no debugging tools
# ------------------------------------------------------------------------------

# Start a netshoot container in the exact same network namespace as the target.
docker run --rm -it --net container:"$CTR" nicolaka/netshoot

# Use netshoot for a specific one-off command instead of a shell.
docker run --rm --net container:"$CTR" nicolaka/netshoot ss -tulpn
docker run --rm --net container:"$CTR" nicolaka/netshoot curl -vk http://127.0.0.1:5000/health

# If you need packet capture inside the target network namespace.
docker run --rm --net container:"$CTR" --cap-add NET_ADMIN --cap-add NET_RAW nicolaka/netshoot tcpdump -nn -i any

# Inspect the Docker network itself, not just the container attachment.
NET_NAME="$(docker inspect --format '{{range $k, $v := .NetworkSettings.Networks}}{{println $k}}{{end}}' "$CTR" 2>/dev/null | head -n1)"
docker network inspect "$NET_NAME" | jq

# ------------------------------------------------------------------------------
# 9. Entering namespaces directly from the host
# ------------------------------------------------------------------------------

# Enter all major namespaces of the target container.
sudo nsenter --target "$PID" --mount --uts --ipc --net --pid

# Run one command in the container namespaces without opening a shell.
sudo nsenter --target "$PID" --mount --uts --ipc --net --pid ip addr
sudo nsenter --target "$PID" --mount --uts --ipc --net --pid hostname
sudo nsenter --target "$PID" --mount --uts --ipc --net --pid ps aux

# If the container has no shell but does have `/proc`, this still works from the host side.
sudo nsenter --target "$PID" --mount --uts --ipc --net --pid sh -c 'pwd || true'

# ------------------------------------------------------------------------------
# 10. Injecting tooling into ultra-minimal containers
# ------------------------------------------------------------------------------

docker cp ./busybox "$CTR":/tmp/busybox
docker exec "$CTR" /tmp/busybox sh

# If the image has no shell but can execute binaries, use busybox for one-shot commands.
docker exec "$CTR" /tmp/busybox ls -lah /
docker exec "$CTR" /tmp/busybox netstat -tulpn

# ------------------------------------------------------------------------------
# 11. Useful commands beyond the original list
# ------------------------------------------------------------------------------

docker stats "$CTR"
docker inspect --format '{{json .State.Health}}' "$CTR" | jq
docker events --filter "container=$CTR"
docker port "$CTR"
docker top "$CTR" -eo pid,ppid,user,args
docker system df
docker container prune
docker inspect "$CTR" | jq '.[0].HostConfig'
docker inspect "$CTR" | jq '.[0].HostConfig | {Memory, MemorySwap, NanoCpus, CpuShares, CpusetCpus, PidsLimit, OomKillDisable, ReadonlyRootfs}'
sudo cat "/proc/$PID/cgroup"
sudo cat "/proc/$PID/status"
docker inspect --format '{{.Config.Hostname}}' "$CTR"
docker inspect --format '{{json .ResolvConfPath}}' "$CTR" | jq -r .
watch -n 1 "docker inspect --format 'status={{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}} started={{.State.StartedAt}}' $CTR"
docker history --no-trunc "$(docker inspect --format '{{.Image}}' "$CTR" 2>/dev/null)" | head -n 20
docker inspect "$CTR" | jq '.[0] | {
  Name,
  Created,
  Path,
  Args,
  State: {Status, Running, ExitCode, StartedAt, FinishedAt, OOMKilled},
  Image,
  Mounts,
  Networks: .NetworkSettings.Networks
}'

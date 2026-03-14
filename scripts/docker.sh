
#!/usr/bin/env bash

# Docker build / run / debug cheat sheet for this project.
# This file is intentionally written as an operator reference, not an executable script.
# Copy the commands you need and set the variables for your target container.

# ------------------------------------------------------------------------------
# Project image lifecycle
# ------------------------------------------------------------------------------

# Build the image from the development Dockerfile.
docker build -t langchain-fastapi-server -f docker/dev.Dockerfile .

# Run the container in detached mode and publish port 5000.
docker run -d \
  -p 5000:5000 \
  --name langchain-fastapi-server \
  --env-file .env.development \
  langchain-fastapi-server:latest

# Follow container logs. Add --tail=200 if the container has been running a while.
docker logs -f langchain-fastapi-server

# Tag for Docker Hub.
docker tag langchain-fastapi-server harmeet10000/langchain-fastapi-server:latest

# Push to Docker Hub.
docker push harmeet10000/langchain-fastapi-server:latest

# For AWS ECR, replace the example below with your ECR repo URI and push that tag.
# docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/langchain-fastapi-server:latest

# ------------------------------------------------------------------------------
# Common variables for debugging
# ------------------------------------------------------------------------------

# Set the target container once and reuse it everywhere below.
CTR=langchain-fastapi-server

# Container ID is often easier to pass into lower-level commands.
CID="$(docker ps -qf "name=^${CTR}$")"

# PID of the container's init process on the host.
PID="$(docker inspect --format '{{.State.Pid}}' "$CTR")"

# Full inspect output as pretty JSON.
docker inspect "$CTR" | jq

# Interactive tree-style JSON inspection for very large inspect output.
docker inspect "$CTR" | jless

# If you have the `js` interactive JSON browser installed, this is also useful.
docker inspect "$CTR" | js

# ------------------------------------------------------------------------------
# Inspecting container configuration without entering the container
# ------------------------------------------------------------------------------

# Show the entire Config object only.
docker inspect --format '{{json .Config}}' "$CTR" | jq

# Show environment variables.
docker inspect --format '{{json .Config.Env}}' "$CTR" | jq -r '.[]'

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
# Minimal / scratch container filesystem access from the host
# ------------------------------------------------------------------------------

# MergedDir gives you the live mounted root filesystem for overlay-based storage.
MERGED_DIR="$(docker inspect --format '{{.GraphDriver.Data.MergedDir}}' "$CTR")"
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
# Network troubleshooting for containers that have no debugging tools
# ------------------------------------------------------------------------------

# Start a netshoot container in the exact same network namespace as the target.
# This is the cleanest way to run ping, dig, curl, tcpdump, ss, or ip for minimal images.
docker run --rm -it --net container:"$CTR" nicolaka/netshoot

# Use netshoot for a specific one-off command instead of a shell.
docker run --rm --net container:"$CTR" nicolaka/netshoot ss -tulpn
docker run --rm --net container:"$CTR" nicolaka/netshoot curl -vk http://127.0.0.1:5000/health

# If you need packet capture inside the target network namespace.
docker run --rm --net container:"$CTR" --cap-add NET_ADMIN --cap-add NET_RAW nicolaka/netshoot tcpdump -nn -i any

# Inspect the Docker network itself, not just the container attachment.
NET_NAME="$(docker inspect --format '{{range $k, $v := .NetworkSettings.Networks}}{{println $k}}{{end}}' "$CTR" | head -n1)"
docker network inspect "$NET_NAME" | jq

# ------------------------------------------------------------------------------
# Entering namespaces directly from the host
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
# Injecting tooling into ultra-minimal containers
# ------------------------------------------------------------------------------

# Copy in a static busybox binary, then use it as a temporary toolbox.
# You need a local static busybox binary for this approach.
docker cp ./busybox "$CTR":/tmp/busybox
docker exec "$CTR" /tmp/busybox sh

# If the image has no shell but can execute binaries, use busybox for one-shot commands.
docker exec "$CTR" /tmp/busybox ls -lah /
docker exec "$CTR" /tmp/busybox netstat -tulpn

# ------------------------------------------------------------------------------
# Useful commands beyond the original list
# ------------------------------------------------------------------------------

# See live CPU, memory, network, block IO, and PIDs usage.
docker stats "$CTR"

# Inspect the health check command and latest probe output.
docker inspect --format '{{json .State.Health}}' "$CTR" | jq

# Watch lifecycle events for a single container in real time.
docker events --filter "container=$CTR"

# Show effective port mapping from the host perspective.
docker port "$CTR"

# Find the top processes running in the container.
docker top "$CTR" -eo pid,ppid,user,args

# Show disk usage by images, containers, build cache, and volumes.
docker system df

# Prune stopped containers only. Safer than broad system prune during incident work.
docker container prune

# Show the exact low-level runtime configuration for mounts, namespaces, and cgroups.
docker inspect "$CTR" | jq '.[0].HostConfig'

# Inspect container resource limits and cgroup settings.
docker inspect "$CTR" | jq '.[0].HostConfig | {Memory, MemorySwap, NanoCpus, CpuShares, CpusetCpus, PidsLimit, OomKillDisable, ReadonlyRootfs}'

# Read cgroup pressure and memory counters from the host through the container PID.
sudo cat "/proc/$PID/cgroup"
sudo cat "/proc/$PID/status"

# Resolve a container's hostname and DNS settings from Docker metadata.
docker inspect --format '{{.Config.Hostname}}' "$CTR"
docker inspect --format '{{json .ResolvConfPath}}' "$CTR" | jq -r .

# Check whether the container is unexpectedly restarting in a loop.
watch -n 1 "docker inspect --format 'status={{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}} started={{.State.StartedAt}}' $CTR"

# Compare image history to understand how the image was built.
docker history --no-trunc "$(docker inspect --format '{{.Image}}' "$CTR")"

# Gather a compact summary fast when joining an incident late.
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

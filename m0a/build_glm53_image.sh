#!/usr/bin/env bash
set -euo pipefail

ROOT=${MEMECHO_ROOT:-/workspace/memecho}
export DOCKER_HOST="unix://$ROOT/runtime/docker.sock"
docker build \
  --tag memecho/m0a-glm53-trace:v3 \
  "$ROOT/m0a/glm_image"

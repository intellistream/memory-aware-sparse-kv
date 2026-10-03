#!/usr/bin/env bash
set -euo pipefail

ROOT=${MEMECHO_ROOT:-/workspace/memecho}
python3 "$ROOT/m0a/download_glm53_tiny.py" \
  --output "$ROOT/models/GLM-5.3-0.6B-A0.4B"

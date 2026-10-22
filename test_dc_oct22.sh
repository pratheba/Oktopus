#!/usr/bin/env bash
# SDF/DCSDD: rigid, wrap, mixed; the full puffer also emits 0087 fused exterior.
# Assumes the Oktopus conda environment and modules are already loaded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python tools/oct22_run.py sdf "$@"

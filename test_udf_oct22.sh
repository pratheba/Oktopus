#!/usr/bin/env bash
# Existing 0084 UDF on EIGHT Puffer SDF PARTS; not UDF on the fused SDF exterior.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec python tools/oct22_run.py udf-parts "$@"

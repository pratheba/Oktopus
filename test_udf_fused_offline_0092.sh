#!/usr/bin/env bash
# 0092: one editable script to reuse SUCCESSFUL SDF and UDF evidence (NO GPU).
set -euo pipefail
cd "$(dirname "$0")"
# ----------------- EDIT HERE -------------------------------------------
SDF_RUN="puffer_oct22_fused_sdf_0089_r128"
UDF_RUN="puffer_oct22_udf_0091_r128"
OUT_RUN="puffer_oct22_fused_udf_0092_offline_r128"
SHAPE="oktopus_9_v1"
# Geometry correspondence tolerances; start with audited defaults.
DISTANCE_EDGE_FACTOR=1.75
NORMAL_DOT=0.55
# ------------------------------------------------------------------------
python tools/cut_fused_sdf_0092.py \
  --sdf-shape "inference/4M/$SDF_RUN/$SHAPE" \
  --udf-shape "inference/4M/$UDF_RUN/_udf_inference_0084/$SHAPE" \
  --out "inference/4M/$OUT_RUN/$SHAPE" \
  --distance-edge-factor "$DISTANCE_EDGE_FACTOR" \
  --normal-dot "$NORMAL_DOT"

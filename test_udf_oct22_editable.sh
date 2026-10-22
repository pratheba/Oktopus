#!/usr/bin/env bash
# 0091: Edit SETTINGS and run: bash test_udf_oct22_editable.sh
# Puffer only; uses existing validated 0081 cut of EIGHT RIGGABLE PARTS.
# This does NOT cut the 0087 global fused SDF exterior.
set -euo pipefail
cd "$(dirname "$0")"

# ====================== EDIT THESE SETTINGS ===========================
SDF_RUN="puffer_oct22_fused_sdf_0089_r128" # your ALREADY COMPLETED SDF output
RESOLUTION=128
OUT="puffer_oct22_udf_0091_r128"    # a NEW run name
UDF_YAML="oct22_adapt_oktopus_puffer_full_udf_optimization.yaml"
UDF_MODEL="config/config_udf_grid_b13d3_oktopus_puffer.yaml"
SHAPE="oktopus_9_v1"
DRY_RUN=0
# ======================================================================

if [[ "$DRY_RUN" != 1 ]]; then
  [[ -d "inference/4M/$SDF_RUN/$SHAPE/riggable_parts" ]] || {
    echo "ERROR: no eight SDF parts for run $SDF_RUN" >&2; exit 2;
  }
  [[ -f "inference/4M/$SDF_RUN/$SHAPE/root_leaves_ownership.json" ]] || {
    echo "ERROR: missing SDF ownership manifest" >&2; exit 2;
  }
  [[ -f "$UDF_MODEL" ]] || { echo "Missing UDF model config: $UDF_MODEL" >&2; exit 2; }
  [[ -f "tools/promote_udf_0081_assembly.py" && -f "tools/udf_stitch_0084.py" ]] || {
    echo "ERROR: 0084 promote/stitch modules missing. Install 0086." >&2; exit 2;
  }
fi

cmd=(python tools/oct22_run.py udf-parts
     --case oktopus_puffer_full --yaml "$UDF_YAML"
     --config "$UDF_MODEL" --shape "$SHAPE"
     --sdf-run "$SDF_RUN" --resolution "$RESOLUTION" --out "$OUT")
[[ "$DRY_RUN" == 1 ]] && cmd+=(--dry-run)
echo '[UDF] stage: inferred eight SDF parts -> curve-local cuts -> GLB -> attachment-stitch attempt'
echo '[UDF] NOT a cut of riggable_fused_sdf_exterior.ply.'
printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

#!/usr/bin/env bash
# 0091: Edit SETTINGS below; run: bash test_dc_oct22_editable.sh
# Calls test_dc.py DIRECTLY. Nothing to pass on the command line.
set -euo pipefail
cd "$(dirname "$0")"

# ====================== EDIT THESE SETTINGS ===========================
CASE="cone"                # puffer | cone | boots_t7 | armadillo_boots | trex_dress | custom
RESOLUTION=128             # 128 first; then 512
OUT="cone_oct22_0091_r128" # unique inference run name
CHECKPOINT="final"
DRY_RUN=0                  # 1 = print only, no inference
EXPECT_FUSED=1             # 1 requires true SDF Boolean exterior, not just GLB assembly

# For CASE=custom, configure these three values:
CUSTOM_YAML="oct22_example_oktopus_cone_fused_sdf.yaml"
CUSTOM_MODEL="config/config_grid_b13d3_oktopus_cone.yaml"
CUSTOM_SHAPE="oktopus_9_v1"

DCSDD_REPO="third_party/dcsdd"
SCRATCH_DIR="logs"
OUTER_ITERS=5
INNER_ITERS=3
# ======================================================================

case "$CASE" in
  puffer)
    YAML="oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml"
    MODEL="config/config_grid_b13d3_oktopus_puffer.yaml"
    SHAPE="oktopus_9_v1" ;;
  cone)
    YAML="oct22_adapt_oktopus_cone_wrap_0093_sdf_optimization.yaml"
    MODEL="config/config_grid_b13d3_oktopus_cone.yaml"
    SHAPE="oktopus_9_v1" ;;
  boots_t7)
    YAML="oct22_example_oktopus_boots_t7_fused_sdf.yaml"
    MODEL="config/config_grid_b13d3_oktopus_boots1.yaml"
    SHAPE="oktopus_9_v1" ;;
  armadillo_boots)
    YAML="oct22_example_armadillo_boots2_selected_fused_sdf.yaml"
    MODEL="" # Enter the actual trained boots_2_v1 model config
    SHAPE="armadillo_13_v1" ;;
  trex_dress)
    YAML="oct22_example_trex_dress_selected_fused_sdf.yaml"
    MODEL="config/config_grid_b13d3_oktopus_dress2.yaml"
    SHAPE="trex_9_v1" ;;
  custom)
    YAML="$CUSTOM_YAML"; MODEL="$CUSTOM_MODEL"; SHAPE="$CUSTOM_SHAPE" ;;
  *) echo "Unknown CASE: $CASE" >&2; exit 2 ;;
esac

YAML_PATH="exp/train/manipulation/$YAML"
if [[ -z "$MODEL" ]]; then
  echo "ERROR: Set the checkpoint-matching MODEL for $CASE at the top of this script." >&2
  exit 2
fi
[[ -f "$YAML_PATH" ]] || { echo "YAML not found: $YAML_PATH" >&2; exit 2; }
if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$MODEL" ]] || { echo "Model config not found: $MODEL" >&2; exit 2; }
  [[ -d "$DCSDD_REPO" ]] || { echo "DCSDD repo not found: $DCSDD_REPO" >&2; exit 2; }
fi

# YAML is the only authority for optimizations and fusion. Do not force
# --dcsdd-bounded-filtering on wrap adapters (the 0091 source guard also
# checks each adaptation individually, including mixed rigid+wrap).
mapfile -t SETTINGS < <(python - "$YAML_PATH" "$SHAPE" "$EXPECT_FUSED" <<'PY'
import sys, yaml
p, shape, require = sys.argv[1:]
d = yaml.safe_load(open(p, encoding='utf-8'))
a = d.get('adaptations', [])
if not a: raise SystemExit('ERROR: missing adaptations')
for i, x in enumerate(a):
    if (bool(x.get('wrap_radius')) == bool(x.get('rigid_radius'))):
        raise SystemExit(f'ERROR: adaptation {i}: select exactly one of wrap_radius and rigid_radius')
    if x.get('target_key','').split('|')[0] != shape:
        raise SystemExit(f'ERROR: adaptation {i} target does not match shape {shape}')
if require == '1' and d.get('sdf_fused_exterior_0087') is not True:
    raise SystemExit('ERROR: EXPECT_FUSED=1 but YAML has no sdf_fused_exterior_0087: true')
print('true' if d.get('dcsdd_bounded_filtering',False) else 'false')
print('true' if d.get('dcsdd_fast_bbox_marking',False) else 'false')
print('true' if d.get('dcsdd_crop_grid',False) else 'false')
print('true' if d.get('sdf_localize_after_gate',False) else 'false')
print('true' if d.get('sdf_fused_exterior_0087',False) else 'false')
print('rigid=' + str(sum(bool(x.get('rigid_radius')) for x in a))
      + ',wrap=' + str(sum(bool(x.get('wrap_radius')) for x in a)))
PY
)
if [[ ${#SETTINGS[@]} -ne 6 ]]; then
  echo "ERROR: unable to read YAML configuration" >&2; exit 2
fi
BOUND="${SETTINGS[0]}"; FAST="${SETTINGS[1]}"; CROP="${SETTINGS[2]}"
AFTER="${SETTINGS[3]}"; FUSED="${SETTINGS[4]}"; MODES="${SETTINGS[5]}"
echo "[SDF] case=$CASE shape=$SHAPE modes=$MODES fused=$FUSED bounded_default=$BOUND"
echo "[SDF] YAML=$YAML_PATH model=$MODEL output=$OUT resolution=$RESOLUTION"

cmd=(python test_dc.py -c "$MODEL" -o "$OUT" -s "$SHAPE" -y "$YAML"
     -r "$RESOLUTION" -ckpt "$CHECKPOINT" -a sdf_dc
     --dcsdd-repo "$DCSDD_REPO" --dcsdd-methods ours
     --dcsdd-outer-iters "$OUTER_ITERS" --dcsdd-inner-iters "$INNER_ITERS"
     --dcsdd-mu 0.5 --dcsdd-dc-weight 0.4 --dcsdd-verbose
     --sdf-grid-memmap-dir "$SCRATCH_DIR" --surface-extraction ours
     --dcsdd-filter-batch-size 262144)
[[ "$BOUND" == true ]] && cmd+=(--dcsdd-bounded-filtering)
[[ "$FAST" == true ]] && cmd+=(--dcsdd-fast-bbox-marking)
[[ "$CROP" == true ]] && cmd+=(--dcsdd-crop-grid)
[[ "$AFTER" == true ]] && cmd+=(--sdf-localize-after-gate)
printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
"${cmd[@]}"
if [[ "$FUSED" == true ]]; then
  base="inference/4M/$OUT/$SHAPE"
  [[ -s "$base/riggable_fused_sdf_exterior.ply" ]] || {
    echo "ERROR: Fused SDF mesh missing: $base/riggable_fused_sdf_exterior.ply" >&2; exit 3;
  }
  [[ -s "$base/riggable_fused_sdf_exterior_report.json" ]] || {
    echo "ERROR: Fused SDF report missing: $base/riggable_fused_sdf_exterior_report.json" >&2; exit 3;
  }
  echo "[SDF SUCCESS] true Boolean exterior: $base/riggable_fused_sdf_exterior.ply"
fi

# 0091: today's reproducible SDF and UDF commands

## Run SDF (puffer, cone, rigid boots, mixed armadillo or T-Rex)

1. **Edit the SETTINGS block at the top** of `test_dc_oct22_editable.sh`.
2. Set `CASE="cone"`, `RESOLUTION=128`, `OUT="cone_oct22_0091_r128"`, `DRY_RUN=1` for a check.
3. Run `bash test_dc_oct22_editable.sh` (no command-line arguments).
4. Set `DRY_RUN=0` and rerun; it calls `test_dc.py` **directly**.
5. If `sdf_fused_exterior_0087: true` in the YAML, the script checks that `riggable_fused_sdf_exterior.ply` and `riggable_fused_sdf_exterior_report.json` were actually created.

The new `oct22_adapt_oktopus_cone_post_sdf_0091_sdf_optimization.yaml` preserves the two wrap cone mappings, merges the two adaptations as one cone part, enables **actual SDF scalar min-union**, and **does not request bounded filtering for ordinary cone wrap**. Cone should no longer hit the bounded-wrap exception. The code additionally disables bounded filtering automatically for any adaptation unsupported by that acceleration (for mixed cases), logging `[sdf bounded 0091]` when it does so. It does not alter geometry or `cut_avatar`.

Supported preset CASEs in the editable script: `puffer`, `cone`, `boots_t7`, `armadillo_boots`, `trex_dress`, `custom`. For `armadillo_boots`, you must supply the checkpoint-matching `boots_2_v1` model config in the script; it is not verified in the uploaded repository. For custom, edit the 3 custom settings. Other Oktopus boots T1..T8: use `custom` and the matching YAML; use the `boots1` model config only where the accessory checkpoint is boots_1_v1.

## Run puffer UDF (eight parts, NOT fused exterior)

1. Edit the SETTINGS block in `test_udf_oct22_editable.sh`.
2. Set `SDF_RUN` to your **completed puffer** SDF output name, `RESOLUTION` to match and `OUT` to a new run name.
3. Run `bash test_udf_oct22_editable.sh`.

The UDF script invokes the already-present `tools/oct22_run.py` to adjust the **input run path** safely and check imports, and then calls the installed `test_udf_sdf_cut_stitch.py`. This 0084 path performs learned UDF inference, 0081 curve-local cut candidates, validates/promotes eight riggable PLYs, assembles the GLB, and **attempts** real stitching. Its stitched PLY may fail on incompatible boundaries; the eight cut PLYs and GLB are kept. **It does not cut the new 0087 SDF fused exterior.** That work remains outstanding.

## What is verified and what is not

Verified: cone config has two wrap adaptations, correct ownership, fused flag and no invalid bounded flag; 0091 bounded eligibility check; Python compile and local SDF shell dry run. Previously verified by actual user run: puffer 0089 fused SDF at 128. Not verified here: cone/GPU geometry, boots/armadillo/T-Rex fused outputs, actual UDF stitching, or UDF-on-fused-surface. Do not claim completion of those pieces yet.

Do **not** enable `cut_avatar: true` as a workaround for this performance-gate exception. It has unrelated geometry semantics.

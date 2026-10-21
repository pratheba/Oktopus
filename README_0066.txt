Oktopus 0066 — UDF cut directly on the final eight root/leaf meshes
===================================================================

What is new
-----------
Opt-in `udf_cut_direct_root_leaves: true` evaluates learned UDFs directly
on the final SDF root/leaf meshes. It does NOT load per-adaptation MC meshes,
transfer cuts from intermediate meshes, reconstruct the avatar, allocate an
unused MCGrid in the UDF runner, or join root/leaves with a Boolean.

For each final part, only adaptation items assigned to that part in the
ownership manifest are queried. Multiple sections of a tentacle contribute
support-aware, normalized UDF scores; unknown UDF evidence never causes a cut.
Far-value-saturated owners abstain rather than cutting everything.

Install: apply the Git patch in the Oktopus repository root.
    git apply --check ~/Downloads/oktopus-0066-udf-final-root-leaves.patch
    git apply ~/Downloads/oktopus-0066-udf-final-root-leaves.patch

This is a Python-only change; rebuilding DCSDD is NOT necessary.

Paired configuration files (both belong in exp/train/manipulation):
    adapt_oktopus_puffer_full_0066_sdf_final_only.yaml
    udf_adapt_oktopus_puffer_final_root_leaves_0066.yaml

1) Produce the eight SDF part meshes at 512 using the new SDF YAML. Use the
   existing test_dc.py command and replace -y and -o, e.g.:
       -y adapt_oktopus_puffer_full_0066_sdf_final_only.yaml \
       -o puffer_0066_final_parts_r512
   The SDF YAML sets export_individual_meshes=false and cut_avatar=false.
   It retains avatar curves/frames for adaptation, not avatar reconstruction.

2) Edit the root/leaf directory in the UDF YAML:
       udf_cut_root_leaves_mesh_dir: inference/4M/puffer_0066_final_parts_r512/oktopus_9_v1/riggable_parts
   Keep the adaptation list in BOTH paired YAMLs identical. The fingerprint
   check fails closed on a mismatch. Use the correct inference/4M path for
   your run's output layout.

3) Run the dedicated UDF-cut runner with your existing UDF MODEL config (not
   the SDF model config) and checkpoint:
       python test_sdf_udf_cut.py \
           -c <your-existing-UDF-model-config> \
           -o puffer_0066_direct_udf_cut_r512 \
           -s oktopus_9_v1 \
           -y udf_adapt_oktopus_puffer_final_root_leaves_0066.yaml \
           -r 512 -ckpt final
   --mc-mesh is not required in this mode. Old modes still validate it.

Expected outputs under the UDF run's shape directory:
   riggable_parts_after_udf_cut/body.ply
   riggable_parts_after_udf_cut/tentacle_*.ply
   riggable_after_udf_cut_UNSTITCHED_parts.ply
   riggable_after_udf_cut.glb
   riggable_after_udf_cut_report.json

Quick diagnostics:
    grep -E '^\[udf cut final (part|root_leaves|saturation)' your_log_file.log

Important validation caveat
---------------------------
The existing UDF workflow used per-item calibration and reference transfer.
This new direct-final mode deliberately uses explicit world-space thresholds:
`udf_cut_seed_world: 0.02`, `udf_cut_grow_world: 0.01`, and a far-plateau
informativeness guard. These choices do NOT guarantee geometric equality to
the legacy cut. Visually compare the 512 cut openings and preservation of
body/tentacle surfaces before running at 2048.

`udf_cut_auto_thresholds.enabled: true` is rejected in this direct-final mode.
Do not disable the ownership-manifest check: it prevents applying the wrong
adaptation's UDF to a different final part.

Validation here
---------------
10 targeted tests passed for the new path plus old combined-direct regressions.
Python syntax validation passed. No real trained-model puffer inference ran in
this environment; SDF/UDF model and mesh quality must be validated on cluster.

Oktopus 0045 — cone zero-angular-follow SDF optimization
========================================================

Builds on 0044 (or on 0042 for cone-only use). Includes:
  test_dc.py
  src/app/agent_3dvec_sdf.py
  src/ngc/pwla_curve/_localize.py
  src/app/sdf_zero_angle_optimization.py    [NEW]
  tests/test_sdf_zero_angular_wrap_0045.py  [NEW]
  exp/train/manipulation/adapt_oktopus_cone_0045_fast.yaml  [NEW]

PERFORMANCE FINDING (from current cone config)
Both cone sections have wrap_angular_follow: 0.0 and the default
multiplicative snug mode. The output radial mapping is
    scale_iso * max(angular_ratio, 1e-8) ** 0.0 = scale_iso.
Thus the computed auto snug scale multiplies an angular ratio that is
raised to zero. The 2 extra expensive localization passes and snug
inferences have no effect on the final coordinates in this config.
0045 deliberately skips these ineffective passes and the angular wrap
interpolations when --sdf-optimize-zero-angular-wrap is requested.

Strict safeguards: additive snug, shared snug field handoff,
nonzero angular follow, existing snug field, attach/free blending,
snug penetration guard, and wrap debug keep their original path.
All other configurations retain the legacy path unless opted in.

Also, for a final combined cone only, the new YAML selects
export_individual_meshes: false. It still evaluates both adaptations,
blends their SDFs within cone_body, and extracts the combined mesh
with --surface-extraction ours. If you need per-section meshes,
use your original YAML or set export_individual_meshes: true.

RUN AFTER INSTALLING FROM REPOSITORY ROOT:
    unzip -o ~/Downloads/oktopus-0045-cone-zero-angular-fast-files.zip

Start with your existing successful 512 command. Replace YAML with
    -y adapt_oktopus_cone_0045_fast.yaml
and add
    --sdf-optimize-zero-angular-wrap
Keep --surface-extraction ours --dcsdd-methods ours and --sdf-localize-after-gate.
Do NOT turn on rigid-only --dcsdd-stream-inference,
--dcsdd-bounded-filtering, or --dcsdd-compact-staging for this combined
wrap-radius cone.

Expected log:
  [sdf_zero_angular_wrap] name= cone_body_1 skip_inert_auto_snug=True
  [sdf_zero_angular_wrap] name= cone_body_2 skip_inert_auto_snug=True
Only one filter_grid_adapt/localize pass per adaptation instead of two.
Only final combined DCSDD mesh is extracted.

TEST:
  PYTHONPATH=src:src/app:src/ngc:tests python -m unittest discover -s tests -p 'test_sdf_zero_angular_wrap_0045.py' -v
  PYTHONPATH=src:src/app:src/ngc:tests python -m unittest discover -s tests -p 'test_sdf_wrap_after_gate_0042.py' -v

Validation: synthetic numerical parity tests passed. The full 512
neural cone run has not been executed by the patch builder. Compare
SDF metrics and final combined PLY (not individual outputs).

IMPORTANT SEMANTIC NOTE: Your existing cone auto snug field has
no effect on coordinates at wrap_angular_follow: 0.0 in the
multiplicative implementation. If you WANT snug to change geometry,
that requires a separate intentional change in mapping semantics
(e.g., nonzero angular follow), not a performance optimization.

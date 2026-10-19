Oktopus 0048 — Audited candidate interval pruning (shared SDF pipeline)
===================================================================
Apply on top of 0047. No change to rigid/wrap mapping, neural model,
support clamp, seam blending, DCSDD, MC, root-leaf ownership, or UDF cut.

Objective
---------
Avoid localizing full-curve broad-phase candidates far from a selected
source interval. The current direct localizer projects on src_0..src_1.
The pruned broad phase uses exactly the previous cylinder-box-marking
algorithm but clips its curve samples to src interval +/- configurable pad.
The actual localizer and accepted-voxel inference order are unchanged.

This narrowing is NOT a proven universal superset of the exact localizer:
curve control smoothing, projection endpoints and geometry can cause a
missing accepted voxel. DO NOT trust an unaudited crop on new geometry.
The optional audit compares the old full candidate union to the reduced
candidate union in bounded sorted-index batches. It applies the original
exact localizer to EVERY excluded candidate, and aborts if ANY would have
been accepted. It runs before model inference. It is expensive ONCE.

Flags (also available as YAML top-level keys)
----------------------------------------------
--sdf-candidate-interval-prune           Enable optional source-interval ROI.
--sdf-candidate-interval-pad 0.10        Pad in [0,1] source-curve units.
--sdf-candidate-interval-audit           Scan excluded candidates with the
                                        original exact localizer; fail closed.

Example cone @512
-----------------
Use adapt_oktopus_cone_0048_interval_audit.yaml with the same 0047 DCSDD
command. The supplied YAML enables the audited mode.
After it reports BOTH pieces with `missing_accepted=0`, and if the exact
model/curve/YAML is unchanged, switch sdf_candidate_interval_audit to false
and benchmark against cone 0047's 356.40s baseline.

Expected acceptance (from previous unpruned 512 cone run):
  cone_body_1: 5,749,845 accepted
  cone_body_2: 9,712,612 accepted
  combined: evaluated=4,167,579 negative=386,655
            sign_changing_edges=73,656 active_cells=73,581
            vertices=73,581 faces=147,312 crop=188^3

If an audit says UNSAFE, stop. Increase pad (e.g. 0.2 or 0.4) and audit
again, or disable pruning. Never ignore this failure or loosen the
localizer to make it pass. Audit is geometry/configuration specific.

Current scope
-------------
Available in the shared 0047 rigid/combined-wrap bounded callback paths.
Puffer's 15-piece root-leaf streaming/composition remains unavailable in
0047 and is NOT unlocked by this patch; its leaf topology is unchanged.
Reuse of localized results between neighboring tentacle adaptations is
planned only after spatial candidate parity passes on actual puffer data.

Validation
----------
PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p \
    'test_sdf_candidate_interval_0048.py' -v
Other targeted tests: test_dcsdd_bounded_filtering.py,
test_sdf_stream_unified_0047.py, test_sdf_combined_wrap_0046.py,
test_sdf_wrap_after_gate_0042.py, test_dcsdd_stream_one_pass_2048.py

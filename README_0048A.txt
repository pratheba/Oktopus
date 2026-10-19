Oktopus 0048a -- tangent-aware audited source-interval candidate envelope
======================================================================
Apply ON TOP of 0048. This hotfix replaces 0048's truncated-cylinder
candidate pruning with a world-space AABB derived from the true 3D source
localizer gate and clipped against the ORIGINAL full-cylinder candidate union.
It does not modify neural inference, rigid/wrap mapping, composition, seam
blending, DCSDD extraction, MC, UDF, or the final SDF acceptance test.

WHY 0048 FAILED FOR CONE
------------------------
0048 shortened the broad-phase cylinder along src_0/src_1. However,
localize_samples() accepts 3D points according to cylinder Y/Z radii AND
calc_x_radius() (tangent axis). In the current PWLA implementation,
calc_x_radius() returns 1.0, so accepted samples can extend significantly
beyond the sliced broad-phase cylinder near interval endpoints. The full
candidate cylinder includes these points, but the sliced cylinder did not.
The audit correctly stopped before inference.

THE HOTFIX
----------
1. Build the same full-curve cylinder used by the original 0047 path.
2. Evaluate center, frame and all THREE localizer acceptance radii across
   the requested source parameter interval.
3. Expand a sampled world-space 3D bounding box to include the accepted
   ellipsoid, including tangent-axis reach and interpolation/grid margins.
4. Intersect the ORIGINAL full-cylinder voxel-box union with that bbox.
5. In audit mode, localize every full-minus-clipped candidate using the
   UNCHANGED localizer and abort on any accepted voxel missed.

The envelope is sampled, NOT a mathematically certified continuous bound.
Always audit first on each changed geometry/configuration/resolution. The
result may prune few or NO candidates because x_radius=1.0 is large.
Do not reduce x_radius to gain speed: that changes the accepted SDF region.

USE (CONE 512)
--------------
Continue using adapt_oktopus_cone_0048_interval_audit.yaml unchanged:

    sdf_candidate_interval_prune: true
    sdf_candidate_interval_audit: true

The old sdf_candidate_interval_pad remains accepted for backward
compatibility but DOES NOT control the new 3D gate envelope. Instead,
look for:

[sdf_candidate_interval_0048a] world_gate_envelope= ...
[sdf_candidate_interval_0048 audit] ... missing_accepted=0 \
    candidate_reduction_pct=...

If the audit aborts again, DO NOT disable the audit while leaving pruning
on. Revert to baseline by setting both values false, and send the logged
world_gate_envelope and missing examples.

Even if audit passes, check that both cone pieces have the SAME accepted
counts (5,749,845 and 9,712,612) and combined SDF/mesh diagnostics as 0047.
An audit is deliberately expensive; time a separate run only after it passes.
If candidate_reduction_pct is small (e.g. < 10%), it is probably better
to disable this ROI optimization and focus on reusable localization across
pieces, which does not change the geometric candidate envelope.

Tests:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_sdf_candidate_interval_0048.py' -v

IMPORTANT: 0048a remains opt-in for the shared rigid / two-piece cone
path. It does not implement mixed puffer's root-leaf streaming or stitching.

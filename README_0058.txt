Oktopus 0058: Conservative sphere rejection inside localization
===============================================================

GOAL
----
Keep the exact 0057 candidate set, projection method, SDF formulas, and
model inputs. Reuse the Euclidean nearest-skeleton-vertex distance already
computed by the KD-tree to skip segment refinement and frame/radius work for
points that mathematically cannot pass the cylinder gate.

SAFETY
------
Only active for the opt-in direct puffer, source interval, k=1, post-gate SDF
path. The original PWLA centerline is piecewise linear and its sampled KD-tree
contains every key point; PWLA frames are orthonormalized; current tangent
radius is identically 1. Accepted voxels must therefore satisfy:

    nearest_vertex_distance <=
        norm * max(1, max(abs(cylinder_key_radius * scale + add)))
        + maximum_sampled_edge_length + rounding_margin.

The inequality is one-sided and cannot reject a truly accepted voxel under
these assumptions. If the tangent-radius implementation is different or
radius data invalid, the optimization disables rejection for that chunk.
It does not change the KD-tree query itself; this targets the post-query
localization workload, not the ~37 s nearest-neighbor query time.

INSTALL
-------
Apply to your current 0057 source (with 0056 already installed):

    git apply --check ~/Downloads/oktopus-0058-localization-sphere.patch
    git apply ~/Downloads/oktopus-0058-localization-sphere.patch

Or unzip the replacement-files archive over the Oktopus root.

Run existing test_dc.py command with the 0058 audit YAML:

    -y adapt_oktopus_puffer_root_leaves_0058_body_audit.yaml \
    -o test_adapt_oktopus_puffer_0058_body_audit_r512

Note: This 0058 audit runs the original localizer on the same candidate chunks
and checks ALL accepted indices and localized avatar/accessory arrays with
exact equality. It will be substantially slower than the previous profile.
0057's earlier candidate-audit success remains a separate prerequisite.

Check logs for:

    [sdf_localize_sphere_0058] enabled=True audit= True
    [sdf_localize_sphere_0058] candidates= 106630916 early_rejected= ...

Expected to remain identical to 0057:
    candidates                   106630916
    accepted                     15466060
    sign-changing edges           191174
    interesting/active cells      189928
    signed volume                 0.143263675

Any mismatch in accepted indices or localized arrays stops the run with
"0058 UNSAFE"; do not proceed to the profile on an audit failure.

AFTER AUDIT SUCCESS
-------------------
Use the 0058 body profile YAML at 512, keeping hardware/CPU allocation fixed:

    -y adapt_oktopus_puffer_root_leaves_0058_body_profile.yaml \
    -o test_adapt_oktopus_puffer_0058_body_profile_r512

It turns off the expensive 0058 inline reference localizations and enables
`sdf_localize_sphere_verified_0058`, for this VERIFIED setup only.
Compare profile to 0057 non-audit baseline: 184.8846 s (body only).

This is an optimization experiment, not an established puffer speedup until
the real 512 run demonstrates it. Full 15-part and 2048 runs are out of scope.

TESTS
-----
    PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_localize_sphere_0058.py' -v

    PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_candidate_piecewise_0057.py' -v

ROLLBACK
--------
Turn `sdf_localize_sphere_0058: false` in YAML or revert the patch. The default
without the flag is the unchanged 0057 localizer.

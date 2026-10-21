Oktopus 0059 — reuse exact avatar localization intermediates
===========================================================

PURPOSE
-------
0058 removes 35.24% of candidates after KD-tree lookup, but 0058 still
recomputes the SAME avatar curve points, full frame matrices, and physical
local coordinates for every accepted voxel inside localize_samples_adapt.

0059 is a different kind of optimization: it does not reject more voxels.
It passes the exact already-computed points/frame/local vectors from the
first localization phase into the subsequent avatar adaptation phase.
It also skips first-pass normalized train coordinates / angles / rho, because
the adaptation layer overwrites all of them before returning anything.

The intermediate keys are private and removed before localize_samples_adapt
returns. The 0059 path is restricted to the direct, one-branch, 0058 SDF
localizer with original avatar frames and train radii. 0059 does not
change curve interpolation math, projection choice, SDF formulas, sampled
coordinates, cylinder thresholds, inference, or contouring.

INSTALL
-------
Apply on top of 0058 with 0057 and earlier dependencies already installed:

  cd /fast/pselvaraju/Oktopus_now
  git apply --check ~/Downloads/oktopus-0059-localization-reuse.patch
  git apply ~/Downloads/oktopus-0059-localization-reuse.patch

Or unzip the replacement-files ZIP over the Oktopus root.

FIRST RUN — EXACT AUDIT
-----------------------
Run the existing 512 body command, replacing YAML and output label with:

  -y adapt_oktopus_puffer_root_leaves_0059_body_audit.yaml \
  -o test_adapt_oktopus_puffer_0059_body_audit_r512

0058 sphere remains enabled, previously audited and in profile mode;
0059 itself has the inline exact parity audit enabled. Each candidate
batch is localized a second time through the 0058 path (with 0059 off).
All accepted original-grid indices, avatar fields, accessory fields,
keys, and NaN positions must be equal. Any mismatch raises
'0058/0059 UNSAFE' and stops the run.

Expected logs:
  [sdf_localize_reuse_0059] enabled=True audit= True
  [sdf_localize_sphere_0058] candidates= 106630916 early_rejected= 37572350
  [sdf_localize_reuse_0059] accepted= 15466060 core_s= ... avatar_s= ...

Expected counts (512 body):
  candidates=106630916, accepted=15466060,
  active_cells=189928, sign_changing_edges=191174,
  signed_volume=0.143263675.

PROFILE ONLY AFTER AUDIT SUCCEEDS
---------------------------------

  -y adapt_oktopus_puffer_root_leaves_0059_body_profile.yaml \
  -o test_adapt_oktopus_puffer_0059_body_profile_r512

Compare against 0058 NON-AUDIT profile 181.7554 s (body only).
The 0059 profiling diagnostics show time spent in the first-pass core
localizer and the subsequent avatar reconstruction, within total
localize_s. No real puffer speedup is claimed until this run succeeds.

LOCAL VALIDATION
----------------
5 new 0059 tests passed, including exact avatar/accessory field parity
for wrap + rigid, scaled/offset radii, empty and all-outside inputs.
The 0058, 0057 and targeted 0054–0056 projection tests passed separately.
A warm 240k-point SYNTHETIC adaptation run took ~0.304s baseline versus
~0.205s with 0059. This is not a prediction of the real puffer speedup.

DEPENDENCIES / ROLLBACK
-----------------------
Requires the working 0058 source and its dependencies.
Set sdf_localize_reuse_0059: false to switch off without code reversal.
Do not turn off the audit for any unverified geometry/configuration.
Do not use the 0059 branch for use_adapt_control_field=true,
use_adapt_train_radius=true, seam outside queries or K>1 projection;
it rejects these unsupported combinations.

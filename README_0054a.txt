Oktopus 0054a -- puffer projection options actually forwarded
==========================================================
Apply on top of 0054, itself on top of 0053.

Problem fixed:
  0054 passed projection settings into extraction_config but NOT the per-piece
  adapt_arg, so the puffer localizer never saw them. 0054's localize_samples()
  also did not accept/forward workers and cache into the interval projector.
  Consequently there were no [sdf_projection_0054] diagnostics and any measured
  speed change could not be attributed to 0054.

Now YAML -> extraction_config -> adapt_arg -> localize_samples_adapt ->
localize_samples -> curve_projection_interval[_candidates].

Default (flags disabled) follows existing projection path. In the opt-in
mode, projection tree reuse and 2-worker KDTree queries preserve accepted-index
and localization-field parity in synthetic tests. Timing tracks actual
interval projection calls and reports per-adaptation projection_s.

Installation (ZIP):
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0054a-projection-forwarding-files.zip
Or install Git patch (NOT both):
  git apply --check ~/Downloads/oktopus-0054a-projection-forwarding.patch
  git apply ~/Downloads/oktopus-0054a-projection-forwarding.patch

First run (same flags/command as 0054):
  -y adapt_oktopus_puffer_root_leaves_0054_body_audit.yaml
  -o test_adapt_oktopus_puffer_0054a_body_audit_r512

Expected:
  [sdf_projection_0054] name= puffer_body_wrap_no_snug workers= 2
      cached_tree= True projection_s= ... projection_calls= ... filter_s= ...

Check the 512 accepted count remains 15,466,060, audit missing_accepted=0,
active cells=189,928, and volume=0.143263675 (reported precision).
No wall-time improvement has been established from these tests. Tests run:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_projection_*0054*.py' -v

This is only the projection cache / workers hotfix. It does not weld
root-leaf surfaces or claim 2048 feasibility.

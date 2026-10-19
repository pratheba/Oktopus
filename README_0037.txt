Oktopus 0037 — Crop-first streamed DCSDD SDF composition
===========================================================
Based on the already-installed Oktopus 0036 source files.
No changes to DCSDD C++, UDF, or adaptation YAML.

Files in archive (paths relative to Oktopus repo root):
  test_dc.py                            replacement
  src/app/agent_3dvec_sdf.py           replacement
  src/app/dcsdd_stream_crop.py         NEW
  tests/test_dcsdd_stream_crop_2048.py NEW
  README_0037.txt                      this file

Install:
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0037-crop-first-stream-files.zip
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_dcsdd_stream_crop_2048.py' -v

First try at r=1024 with the Boot 1-only YAML and your exact 0036 flags,
plus --dcsdd-stream-crop-first. Keep --dcsdd-stream-inference and
--dcsdd-crop-grid and set a NEW output directory.

What changes:
  pass 0: count accepted voxel samples
  pass 1: neural inference + exact global support-profile calibration
  pass 2: recompute final SDF in bounded chunks, locate exact negative bbox
  pass 3: recompute final SDF in bounded chunks, write only into the padded
          negative-bbox cube (as ordinary RAM grid when <=100M voxels)
  extraction: use that crop directly with identical origin/spacing; do not
              scan or initialize the original dense 1025^3 / 2049^3 grid
  progress lines now print during bbox and cropped-write passes.

This introduces ONE EXTRA localization pass versus 0036. It removes
full-volume clear_grid() writes and full-grid scans and keeps output SDF
values/marks identical to the old cropped field in local parity tests.
Benefits at 1024/2048 are still unmeasured; no wall-clock guarantee.

Restrictions:
  Only independent direct rigid-radius boots, dcsdd-method=ours,
  dcsdd-crop-grid, stream-inference, memmap-dir, existing 0036 restrictions.
  Crop-first is optional; without the new flag the 0036 path is unchanged.

Success criteria:
  Look for crop=199^3 start=[557,375,580] on the prior 1024 Boot 1.
  Check evaluated=7,226,684; negative=377,659; interesting_cells=68,696;
  V=68,696 F=138,456 and compare meshes before claiming full parity.
  Check `crop_bbox_pass_s`, `compose_grid_s`, and total runtime versus 0036.

OKTOPUS 0036 — PREDICTION-STREAMED RIGID BOOT EXPERIMENT
========================================================

Base required: 0035 (which includes 0031-0034). Extract this ZIP at the
Oktopus repository root; it contains full replacement files (not a patch).

Files:
  test_dc.py                              (updated CLI flag)
  src/app/agent_3dvec_sdf.py              (optional streaming route)
  src/app/dcsdd_stream_boot.py           (NEW: batch handling, global support)
  src/ngc/curve_handle.py                 (optional callback with no staging)
  tests/test_dcsdd_stream_2048.py         (NEW parity tests)

Enable with --dcsdd-stream-inference. Keep --dcsdd-bounded-filtering,
--dcsdd-crop-grid, --dcsdd-fast-bbox-marking, and --sdf-grid-memmap-dir.
Do NOT pass --dcsdd-compact-staging; the streamed path supersedes it.

SUPPORTED: independent direct rigid-radius Boot 1, no tiled detail, no
avatar carve, no snug, no seams/blend groups, no offsets, no global feature
fields, and export_individual_meshes=true. Unsupported cases explicitly
raise ValueError instead of silently modifying geometry.

The clamp_pred_sdf_by_support() algorithm calculates two global 48-bin
quantile profiles, one for dist and one for dist_base. Running this separately
per batch would change the mesh. Therefore 0036 uses THREE passes through
the localization function but only ONE neural inference pass:
    1. Count accepted candidates; no retained candidate arrays.
    2. Re-localize and infer using exactly the same global batch partition as
       _inference_full_vals() from 0035. Stage only dist, dist_base and
       dist_detail as three contiguous scalar prediction files. Collect
       near-surface rho values to compute the global quantiles exactly.
    3. Re-localize (NO inference), clamp each chunk against the global
       quantile profiles, recompose the same base+gate*detail formula and
       update the disk-backed grid at the original voxel indices.
Then extract DCSDD exactly as before. No C++ or UDF changes.

This reduces staged *localization records* from 3.791 GB at 1024 to zero,
replacing them with THREE smaller scalar prediction arrays. The exact
prediction storage depends on actual network output dtypes: for 47,383,570
float32 rows it should be ~0.569 GB. The extra localization passes have a
runtime cost. Speedup is NOT guaranteed until timed on the cluster.

Test:
  PYTHONPATH=src:src/ngc python -m unittest discover \
    -s tests -p 'test_dcsdd_stream_2048.py' -v

Run Boot1-only, resolution 1024, in a NEW output directory. Example flags:
  -r 1024 -y adapt_oktopus_boots_t7_boot1_only.yaml \
    -o test_adapt_oktopus_boot1_stream_1024 \
    -a sdf_dc --surface-extraction ours --dcsdd-methods ours \
    --dcsdd-bounded-filtering --dcsdd-stream-inference \
    --dcsdd-filter-batch-size 262144 --dcsdd-fast-bbox-marking \
    --dcsdd-crop-grid --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs

IMPORTANT: Compare output PLY and sign-change counts with your successful
0035 1024 reference. No full trained-model or 2048 production validation is
possible in the local test environment. Leave existing 0035 installed as a
fallback (simply omit --dcsdd-stream-inference).

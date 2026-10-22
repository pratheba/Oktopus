# 0100: Direct learned-UDF query on the true fused SDF mesh (experimental)

Unlike 0092, this evaluates the learned UDF **again at fused SDF face samples**. It does not reuse old 0081 cut decisions, the 0079 eight-face gate, or 0081 expansion. The old 0092 NPZ is used *only to map each fused face to its source part*; it must be from the exact same SDF run.

No claim that missing UDF evidence means absent surface. The three decisions are:

- **Supported**: at least one relevant valid UDF owner gives score <= grow threshold.
- **Unsupported**: all relevant UDF owners were valid, and every score >= seed threshold (but below far sentinel).
- **Unknown**: no valid query, no owner, incomplete owner coverage, or score in the ambiguous interval. Unknown stays in mesh. For the 1024 bottom, this may still retain the cap; check the `missing_valid_model_evidence` count.

## Run

On the cluster, on top of the 0092/0081 source:

```bash
python tools/udf_0100_run_direct.py \
  --sdf-run adapt_oktopus_puffer_sdf_r1024 \
  --udf-run adapt_oktopus_puffer_udf_r1024 \
  --out puffer_udf_direct_fused_0100_r1024 \
  --resolution 1024 \
  --parts body \
  --ckpt final
```

Use the **same UDF checkpoint** as the successful 0081 run (replace `latest` if different). This re-runs UDF inference on the fused surface, *not* SDF inference. It leaves original SDF and UDF run directories untouched.

Result:

```
inference/4M/puffer_udf_direct_fused_0100_r1024/oktopus_9_v1/udf_direct_fused_0100/
  fused_direct_kept.ply
  fused_direct_removed.ply           # if any
  fused_direct_unknown.ply
  body_removed.ply                 # when testing body
  body_unknown.ply                 # body-region unknown evidence
  consistency.npz
  report.json
```

Compare with `riggable_fused_sdf_exterior_after_udf_cut.ply` from 0092, **not** the unstitched GLB. `--parts body` deliberately keeps other parts unchanged for the first trial. The `parts=all` mode exists only as an experiment; non-body calibrations require validation (see `--allow-uncalibrated-nonbody`).

**Safety:** direct UDF scores after adaptation are distance-*like* but not necessarily metrically calibrated. End-cap samples outside adaptation support remain unknown; 0100 does not fabricate absence from such invalid queries. This patch is an honest independent test of whether a direct model query solves the bottom failure; it is not yet a guaranteed cap-removal fix. No topology cleanup, smoothing, stitching or re-meshing is done.

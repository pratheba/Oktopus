# 0102 — Fix `--parts all` in the 0100 direct fused UDF experiment

This patch **does not modify any UDF classifier or cutting logic**.

- 0100 previously defaulted to `--parts body`, even when `--parts` was omitted.
- It passed `['all']` to the runtime for `--parts all`, but the runtime expects the literal string `all`.
- Now omitted `--parts` means `all`, and `--parts all` passes the correct runtime sentinel.
- Non-body direct UDF thresholds are still uncalibrated. The runner requires **explicit** `--allow-uncalibrated-nonbody` for an all-parts preview, and fails *before launching the GPU job* if it is missing.

Example from the repository root:

```bash
python tools/udf_0100_run_direct.py \
  --sdf-run adapt_oktopus_puffer_sdf_r1024 \
  --udf-run adapt_oktopus_puffer_udf_r1024 \
  --out puffer_udf_direct_fused_0102_all_r1024 \
  --resolution 1024 \
  --parts all \
  --allow-uncalibrated-nonbody \
  --ckpt final
```

This is a *diagnostic* all-parts run, **not a production-safe calibrated cut**. Each tentacle's actual direct-UDF score/threshold scale must be checked before approving removed faces. Do not confuse the diagnostic `unknown` mesh with a removal mask. Original runs are untouched; always specify a fresh `--out`.

Dry-run command and checks:

```bash
python tools/udf_0100_run_direct.py --sdf-run adapt_oktopus_puffer_sdf_r1024 \
  --udf-run adapt_oktopus_puffer_udf_r1024 --out puffer_udf_direct_fused_0102_all_r1024 \
  --parts all --allow-uncalibrated-nonbody --dry-run
python -m unittest discover -s tests -p test_udf_0102_all_parts_cli.py -v
```

Expected runtime log has one `[udf direct fused 0100] <part> owner= ...` line per selected UDF owner, not only `body`; `report.json` should include all eight part names.

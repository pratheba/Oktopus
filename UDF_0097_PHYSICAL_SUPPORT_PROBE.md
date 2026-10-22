# 0097 — Read-only physical-support probe for UDF cap evidence

## Purpose

Test the hypothesis that 1024 cap-edge UDF evidence is fragmented by sampling / discrete per-face connectivity, while the body cap surface itself remains connected. This experiment requires **only previously saved 0074/0079/0081 evidence and original SDF part PLYs**. It does not rerun inference and it does not modify the cut or perform a high-resolution mesh deletion.

For each run, the script compares the failing cap-edge band `s=[0.885,0.900)` with the previously working body band `s=[0.400,0.450)`. It tests simulated support obtained by traversing short paths **along shared mesh edges** from saved high-score faces. Paths are constrained by normal and curve-coordinate continuity. This is *physical-radius support*, not a new learned UDF score or an approved production cutting rule.

## Run

```bash
cd /fast/pselvaraju/Oktopus_now
python tools/udf_0097_physical_support_probe.py \
  --old inference/4M/puffer_0081_curve_local_preview_r512 \
  --new inference/4M/adapt_oktopus_puffer_udf_r1024 \
  --radii 0,0.002,0.004,0.006 \
  --out /tmp/puffer_0097_physical_support.json \
  | tee /tmp/puffer_0097_physical_support.txt
```

If saved `input` PLY locations inside the run reports do not exist on the current machine, pass `--old-mesh <PLY>` and `--new-mesh <PLY>` pointing to **the exact original meshes used in the corresponding UDF runs**.

The zero-radius row is the baseline for the *cropped arc window*. It can differ from the original full-mesh 0079 vote result because components may cross the window boundary; for exact accepted counts use the separately displayed `stored_votes` value.

## Interpret

- If the 1024 cap gains many accepted strong-score faces with a very small physical radius, while the working control barely changes, there is evidence that isolated samples/fragmented score connectivity drive this regression.
- If no small radius reconnects the score support, investigate the source SDF surface or UDF query samples instead; do not increase the radius blindly.
- If the control changes substantially or short paths traverse undesired geometry, the proposed physical-neighborhood operation is unsafe as-is.
- This probe **does not establish** that the original per-triangle score generation is the root cause. It deliberately tests what happens if valid low-score faces lying along short geodesic paths are allowed to bridge high-score islands, leaving the original score and seed thresholds unchanged.

Do **not** use its results as a production cut or convert accepted areas to removals without evaluating the new boundary, body opening, and tentacle sidewalls. Existing 0081/0092 outputs are untouched.

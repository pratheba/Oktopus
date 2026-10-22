# 0087 | Real SDF Boolean exterior with eight riggable parts

The uploaded 2026-10-22 Oktopus source writes eight independent root/leaf parts.
The 0084 ``STITCHED.ply`` is only a boundary-loop bridge; it is **not** a
Boolean union and may contain interior overlapping surfaces. A scalar-field
Boolean union is `min(sdf_body, sdf_t1, ..., sdf_t8)` after each blend group
is finalized. This patch exports that actual zero-level surface **in addition**
to the eight untouched riggable SDF parts. It does NOT change rigid/wrap
adaptation parameters, seam corrections or `cut_avatar: false`.

## Install

Apply 0087 on the uploaded Oct-22 repository; 0084 and 0085 remain installed.

```bash
git apply --check ~/Downloads/oktopus-0087-true-fused-sdf.patch
git apply ~/Downloads/oktopus-0087-true-fused-sdf.patch
```

## SDF run | 512 first

One SDF call; outputs all eight original riggable parts and the union.

```bash
python test_dc.py \
  -c config/config_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0087_fused_sdf_r512 -s oktopus_9_v1 \
  -y adapt_oktopus_puffer_full_0087_fused_exterior.yaml \
  -r 512 -ckpt final -a sdf_dc \
  --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
  --dcsdd-methods ours --dcsdd-outer-iters 5 --dcsdd-inner-iters 3 \
  --dcsdd-mu 0.5 --dcsdd-dc-weight 0.4 \
  --dcsdd-verbose --dcsdd-crop-grid --dcsdd-fast-bbox-marking \
  --dcsdd-bounded-filtering --dcsdd-filter-batch-size 262144 \
  --sdf-localize-after-gate --surface-extraction ours \
  --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs
```

Outputs under `inference/4M/puffer_0087_fused_sdf_r512/oktopus_9_v1/`:

- `riggable_parts/body.ply`, `tentacle_t1.ply` ... `tentacle_t8.ply`: unchanged separate ownership.
- `riggable_fused_sdf_exterior.ply`: **true global scalar min-union**, not a triangle concatenation.
- `riggable_fused_sdf_exterior_report.json`: connectivity, open edges, nonmanifold edges and watertight diagnostic.
- `root_leaves_ownership.json`, GLB and unstitched parts: riggable source assets.
- Existing `combined_ours_*.ply`: same fused geometry under the native DCSDD filename.

### Validation criteria

First inspect `riggable_fused_sdf_exterior.ply` by itself, *without displaying
other mesh nodes*. It must have no former T3 cap or internal overlapping
body/leaf geometry; visually verify there is no spurious outside component.
Also review `riggable_fused_sdf_exterior_report.json`: a union is not necessarily
one connected component if the physical SDF volumes do not intersect.

### What this does NOT do

- It does **not** run learned UDF on the fused surface. The 0084 UDF runner
  still cuts the eight separate parts. Running that runner unchanged does not
  produce a UDF-cut Boolean exterior, even if 0087 fused SDF exists.
- It does **not** guarantee closed/manifold output in areas where the original
  SDF support was never queried; a scalar union cannot invent missing support.
- It does **not** apply avatar Boolean; `cut_avatar: false` stays unchanged.
- Enabling fused union has an extra full 512³ disk-backed grid and extraction;
  the original 0067 timing (876.82 seconds) does NOT apply.

**Next integration step:** apply learned UDF ownership and curve-local cut
on the *fused mesh itself*, using per-part ownership as a spatial prior, then
validate openings. Do not rename the 0084 boundary bridge as a fused shell.

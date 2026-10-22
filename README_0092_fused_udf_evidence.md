# 0092 — Fused-SDF exterior, evidence-guided UDF cuts (puffer)

**Why:** The 0084 `STITCHED.ply` joins separate meshes via seven loops but is **not** a Boolean union and fails because the body only has five separate loops. 0089 SDF **already** produced a real scalar-field-unioned exterior: `riggable_fused_sdf_exterior.ply`.

0092 never calls the seven-loop stitcher when `udf_fused_exterior_0092.enabled: true`. It uses the **genuine 0089 fused surface** and projects existing **learned UDF evidence from 0081** onto that fused triangulation. It does **not** evaluate the learned UDF a second time on the fused mesh. It is an evidence-transfer approach with diagnostic safeguards, not the fully direct fused-UDF algorithm. It does not close holes, make the mesh watertight or produce a rigged single topology automatically.

## Fastest test: no GPU, no inference, no SDF rerun

You have both completed stage directories already. After applying the patch, edit the **top** of `test_udf_fused_offline_0092.sh` only if your names differ:

```bash
SDF_RUN="puffer_oct22_fused_sdf_0089_r128"
UDF_RUN="puffer_oct22_udf_0091_r128"
OUT_RUN="puffer_oct22_fused_udf_0092_offline_r128"
```

Then run:

```bash
bash test_udf_fused_offline_0092.sh
```

This reads:

- `inference/4M/$SDF_RUN/oktopus_9_v1/riggable_fused_sdf_exterior.ply`
- `inference/4M/$SDF_RUN/oktopus_9_v1/riggable_parts/*.ply`
- `inference/4M/$UDF_RUN/_udf_inference_0084/oktopus_9_v1/udf_cut_curve_local_0081/<part>/locality.npz`

And writes one new directory:

```text
inference/4M/$OUT_RUN/oktopus_9_v1/
  riggable_fused_sdf_exterior_after_udf_cut.ply
  riggable_fused_sdf_exterior_after_udf_cut_report.json
  fused_udf_removed_faces.ply               (if cut)
  fused_udf_transfer_evidence_0092.npz
  fused_udf_detached_fragments.ply          (if detached)
```

Open the output PLY **alone** (hide original SDF and all eight riggable parts). Check the body openings and T3/T8 ends. Inspect `parts.tentacle_t8.zero_seed_audit`, `unresolved_zero_seed_parts`, `transfer.unmatched_faces`, and `is_watertight` in the JSON report. No final geometry claim is made until checked.

## One-command NEW inference path

The patch includes:

`exp/train/manipulation/oct22_adapt_oktopus_puffer_full_udf_fused_0092.yaml`

This copies the established 0091 UDF YAML **without changing the 15 adaptations or 0081 thresholds**, and adds:

```yaml
udf_fused_exterior_0092:
  enabled: true
  distance_edge_factor: 1.75
  normal_dot: 0.55
  zero_seed_cap_fallback: true
```

The editable `test_udf_oct22_editable.sh` now points to this YAML. On a **new output name**, it will: run learned UDF inference on the eight original 0089 riggable parts, save/assemble their individual cut PLYs and GLB, then project removal decisions onto the **true fused-SDF mesh**. It **skips** 0084 stitching.

```bash
bash test_udf_oct22_editable.sh
```

**Do not run inference again now just to test this.** Use the offline script above on the already finished SDF/UDF runs first.

## Safety and known limitations

- Matching requires nearest-point distance to source **triangles** and aligned face normals. Unmatched faces are **retained**. Curved self-near surfaces can still make correspondence ambiguous; inspect the removed-faces PLY.
- When a part has no 0081 seeds (e.g. T8), a candidate connected cap component is deleted only if it is narrow and terminal along the original curve and has coherent, near-axial normals. The zero-seed classifier can abstain; if so the cap remains, with the reason recorded. It never blindly cuts all unknown faces.
- The global fused SDF union does not necessarily make disconnected physical solids touch. An open UDF cut should not be watertight, and nonmanifold geometry may remain; the JSON reports actual topology.
- The present 0084 inference/promotion code still uses the puffer eight-part plan. The **offline projection module** reads part names from the SDF manifest, but the **full inference wrapper** is not yet generic for boots/armadillo/dress.
- Existing SDF outputs, 0091 UDF evidence and individual riggable assets are not modified by the offline script.

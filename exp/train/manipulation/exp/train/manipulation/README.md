# Oktopus adaptation configs — cleaned selection (24 November 2026)

This directory is a **curated replacement** for the 198 root YAML files in the
uploaded `manipulation.zip`. The original ZIP is your unmodified experiment archive.
Do not delete it: the excluded Oct22/Oct23 trials still document tuning history.

## Proven working Oktopus references (copied byte-for-byte)

- `oct23_adapt_oktopus_boots_t1_stream_2048.yaml`
- `oct22_adapt_oktopus_boots_t7_boot1_only_sdf_optimization.yaml`
- `oct23_adapt_oktopus_cone_wrap_canonical_world_0103.yaml`
- `oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml`

These four are your identified working experiments, **not** proof that their
performance flags safely transfer to T-rex or armadillo.

## New SDF adaptation starting points

| File | Mappings | Basis / confidence |
|---|---:|---|
| `nov24_adapt_trex_dress_sdf.yaml` | 2 | 2 active records, top+bottom. Source dress_2_v2 based on newly preprocessed data. OCT22 source was dress_2_v1; check the new checkpoint. |
| `nov24_adapt_armadillo_dress_sdf.yaml` | 2 | 2 active records, top+bottom, with separate cache keys. Alternative bottom-only tests archived in original ZIP. |
| `nov24_adapt_trex_puffer_sdf.yaml` | 4 | 4 records from oct22 Trex-v2 puffer, retargeted to Trex-v1 after comparison with uploaded v1 part names; scale/placement needs visual confirmation. |
| `nov24_adapt_armadillo_puffer_sdf.yaml` | 4 | 4 new puffer_3_v1 source mappings using Armadillo long_puffer geometry; especially neck must be visually tuned. |
| `nov24_adapt_trex_boots_sdf.yaml` | 2 | NEW and untested: boots_1_v1 boot mapped to both rear legs of Trex-v1 with neutral scale/rotation. Not a validated fit. |
| `nov24_adapt_armadillo_boots_sdf.yaml` | 2 | Only shaft test #0 and foot #3 retained from oct22 4-record experiment; assumes boot_2_v1 source model and handle are available. |

**Important corrections to the OCT22 imports:**

1. The original T-rex dress YAML ran six mutually competing top/bottom
   variants; the new file uses one top and one bottom.
2. Armadillo dress previously contained four alternative bottom settings with
   the same `cache_as`; the new file uses one top and one bottom with distinct cache keys.
3. Armadillo boots previously ran three shaft variants plus a foot; the new
   file runs one shaft and one foot.
4. The existing T-rex puffer experiment targeted `trex_9_v2`, while your recent
   T-rex dataset is `trex_9_v1`. The new file uses v1 curves, but exact
   geometric equivalence has NOT been demonstrated.
5. Original dress configs refer to `dress_2_v1`. New configs refer to the
   newly preprocessed **`dress_2_v2`**. Verify the checkpoint/model actually
   contains those curve embeddings. Preprocessing alone does not guarantee it.
6. The Oct22 generic SDF optimization flags are retained only where they were
   general and conservative; OCTOPUS-specific sparse-puffer/0087 fused settings
   and 0093 post-blend Boolean were **not** copied into these new avatars.
7. For each multi-part outfit, `combine_adaptations: true` intentionally
   extracts a merged SDF in addition to independent adaptation outputs.
   Unlike the old Oct22 snapshots (which lacked this flag), this does not
   silently stop after exporting separate pieces. Examine merged seams.
8. `cut_avatar` remains false in the new files, consistent with the OCT22
   SDF/UDF separation. This does not by itself prevent body collisions.

## UDF: what the accompanying drafts are and are not

`udf_drafts_NOT_FOR_0084_RUNNER/` contains *matching adaptation records* for
independent UDF experiments. The old `test_udf_sdf_cut_stitch.py` 0084 runner
expects the **OktopUS puffer 15-adaptation/8-part manifest**, so the drafts
are NOT validated cut/stitch configs for new outfits. The later 0107 direct-UDF
inference method uses the SDF output/ownership data; it does not require a
separate adaptation YAML for each UDF cut. Do not assume a standalone UDF draft
will successfully remove closure faces.

## Preflight validation

```bash
python validate_manipulation.py .
python validate_manipulation.py . --data-root /lustre/fast/fast/pselvaraju/Oktopus_now/Pack3MDataset
```

The optional data-root step checks whether expected `std_handle.npz` curve
names exist for each source and avatar. **It cannot validate checkpoint model
embeddings, learned SDF quality, placement, fusion seams, or final UDF cutting.**

Example SDF command (run from repository root; use your own trained config):

```bash
python test.py -c "$SDF_TRAINING_CONFIG" \
  -o trex_dress_first_test -s trex_9_v1 \
  -y nov24_adapt_trex_dress_sdf.yaml -r 128 -ckpt final
```

Set `SDF_TRAINING_CONFIG` to your **actual joint training config** whose
model/checkpoint includes both the avatar and the target accessory. In particular
`dress_2_v2` must be present; its preprocessed `sdf_samples.pkl` alone is not
sufficient. Running at `-r 128` is only a geometry smoke test.

## Source provenance

Source settings were read from:
- `adapt_trex_dress.yaml`
- `adapt_armadillo_dress.yaml`
- `adapt_trex2_puffer.yaml`
- `adapt_armadillo_long_puffer.yaml`
- `adapt_armadillo_boots2.yaml`
- `oct22_adapt_oktopus_boots_t7_boot1_only_sdf_optimization.yaml`
- the common safe Oct22 SDF flag set from `oct22_adapt_trex_dress_sdf_optimization.yaml`

No experiment, mesh, inference or model evaluation was executed on the six new
YAMLs. Treat all six as **provisional** until run on your `current` branch.

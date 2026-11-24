# Original manipulation ZIP — audit notes

The uploaded original folder contains **196 root-level `.yaml` files** plus YAMLs in subfolders. This cleaned copy has **10 root-level `.yaml` files**: four user-identified Oktopus references and six new target SDF starters. The full original ZIP remains the historical archive.

Original root YAML classification: 187 valid adaptation YAMLs, 8 separate-operation YAMLs, and 1 malformed file.

The malformed historical file is `udf_adapt_oktopus_puffer_bkup.yaml` (YAML parsing error near line 26); it is excluded. The eight separate-operation YAMLs (`mix_`, `stretch_`, `transform_`, `adapt_boots.yaml`) are not adaptation-profile errors and are deliberately excluded from this focused replacement.

**Do not classify repeated target keys as globally invalid.** 132 adaptation YAMLs contain repeated targets; some are intentional multi-section adaptations (such as the working Oktopus puffer and cone), while others are alternative parameter experiments (such as the T-rex dress and Armadillo shaft). The six new profiles have been curated individually rather than deduplicating everything mechanically.

### What static verification checked

- All cleaned root YAMLs parse as adaptation lists or mapping + `adaptations`.
- No duplicate YAML mapping keys, empty/invalid `target_key` or `accessory_key` strings, conflicting rigid/wrap flags, or zero-length source/target curve intervals.
- Multi-adaptation SDF outputs explicitly set `combine_adaptations: true` to enable combined extraction (unlike the old generic OCT22 import default).
- OCT22 generic optimization keys accepted by the uploaded October 2026 agent's config parser; no OCTOPUS-only 8-part root-leaf stitching graph was invented for the new avatars.
- Each new UDF draft has precisely the same adaptation records as its new SDF counterpart.
- All four user-identified working reference YAML files are byte-for-byte unchanged.

### Limits

I cannot verify inferences, source/accessory checkpoint embeddings, spatial fit, seam topology, Boolean operations, or UDF cut results without the **current** repository code, actual handle NPZ files and model checkpoints. The supplied Oktopus code snapshot used for the interface audit is from **October 22, 2026**, whereas your active branch has newer changes. Run `validate_manipulation.py --data-root ...` on the cluster, then low-resolution geometry tests, before launching 2048-grid work.

Oktopus 0067 — full 512 puffer SDF → final-parts UDF handoff
========================================================

This bundle contains TWO paired configs. Install the 0066 Python patch first.
Copy both YAMLs to exp/train/manipulation/ in your Oktopus repo.

Problem solved:
    0058 sphere and 0059 reuse were activated globally in 0066.
    Those guards require the tested direct puffer WRAP staging path.
    Adaptation 2/15 is RIGID and must not use the body-only path.
    0062 spatial filtering also requires 0058, so disable both together.
    0057 and 0048 pruning were also only audited for the body, so leave
    these OFF by default for other parts. Base 0053 bounded direct streaming,
    0054/55 projection refinements, 0056 zero-copy and root-leaf group blending remain.

SDF run:
    -y adapt_oktopus_puffer_full_0067_sdf_safe.yaml \
    -o puffer_0067_final_parts_r512
    Keep all other previously working test_dc.py arguments and -r 512.
    Export only final 8 meshes. No SDF cut_avatar. Do not add stop_after.

Confirm the root-leaf output has:
    inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1/riggable_parts/body.ply
    ... tentacle_t1.ply ... tentacle_t8.ply (the seven configured leaves)
    inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1/root_leaves_ownership.json

UDF run (use YOUR working UDF model config/checkpoint):
    python test_sdf_udf_cut.py \
      -c <YOUR_UDF_MODEL_CONFIG> \
      -o puffer_0067_direct_udf_r512 \
      -s oktopus_9_v1 \
      -y udf_adapt_oktopus_puffer_0067_final.yaml \
      -r 512 -ckpt final

    - The UDF YAML already points at the SDF run's riggable_parts folder.
    - Do not alter the UDF adaptation list independently: manifest fingerprint
      is exact (including six rigid opt-out flags).
    - The direct-final cutter uses explicit world-space thresholds 0.02/0.01.
      These thresholds are UNVALIDATED on actual puffer outputs. Verify it
      removes intended cap faces, does not carve good surfaces, and preserves
      original holes and unrelated tentacles before running 2048.
    - The eight outputs are SEPARATE/UNSTITCHED, not manifold-unioned.

Boots / cone:
    Do not copy these puffer-specific flags into boots or cone configurations.
    Their localization and cutting paths differ; validate each independently.

Owner fingerprint (matched SDF & UDF): 50107ce9192c274da40c68ed95576c6471091c9a69738a78630eac0a2cdf0976

Oktopus 0046 — experimental combined wrap-cone streaming
==========================================================
Apply on top of 0045 (which itself includes prior patches 0032–0044).

SCOPE — deliberately narrow and fail-closed:
  Exactly TWO direct wrap-radius pieces, same target, one post-SDF seam_blend
  group, wrap_angular_follow=0, mathematically inert multiplicative auto-snug,
  support clamp enabled, NO avatar carve, no tiled detail or geometry-altering
  offset, combined output and no individual mesh export.
  Cone 0045 YAML meets these requirements. Puffer DOES NOT yet.
  Unmodified rigid-boot 0040/0041 and normal cone paths stay available.

Files:
  test_dc.py
  src/app/agent_3dvec_sdf.py
  src/app/sdf_stream_combined_wrap.py        NEW
  src/ngc/curve_handle.py
  src/ngc/pwla_curve/_localize.py
  tests/test_sdf_combined_wrap_0046.py       NEW
  exp/train/manipulation/adapt_oktopus_cone_0046_stream.yaml NEW

Behavior:
  Each cone piece: traverse broad-phase candidates once using bounded
  iterator, localize exactly once, infer in fixed accepted-point batches,
  save only three predictions, replay metadata and avatar source coordinates.
  Calibrate support rho globally PER PIECE (same quantile method). Recompose
  exactly the same base+detail+support field (subject to pointwise batch
  numerical differences). Merge the two sparse fields by the original
  post-SDF overlap crossfade using the right-hand avatar source coordinate.
  Find final negative bbox and compose only its padded cube. Extract with
  DCSDD 'ours'; do not allocate/clear full 513^3/2049^3 accumulators.

RUN 512, use a NEW output folder, e.g.:
  python test_dc.py \
    -c config/config_grid_b13d3_oktopus_cone.yaml \
    -o test_adapt_oktopus_cone_dcsdd_r512_0046 \
    -s oktopus_9_v1 \
    -y adapt_oktopus_cone_0046_stream.yaml \
    -r 512 -ckpt final -a sdf_dc \
    --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
    --dcsdd-methods ours \
    --dcsdd-outer-iters 5 --dcsdd-inner-iters 3 \
    --dcsdd-mu 0.5 --dcsdd-dc-weight 0.4 \
    --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
    --dcsdd-stream-replay-dir /fast/pselvaraju/Oktopus_now/logs \
    --surface-extraction ours --dcsdd-verbose

The YAML enables --sdf-stream-combined-wrap, --sdf-localize-after-gate,
--sdf-optimize-zero-angular-wrap, and --dcsdd-crop-grid.
Do NOT add the old rigid-boot flags --dcsdd-stream-inference,
--dcsdd-stream-one-pass, --dcsdd-stream-crop-first or
--dcsdd-stream-replay-metadata. The combined path handles them internally.

Useful logs:
  [sdf_stream_combined_wrap] piece= 0 ... accepted= ...
  [sdf_stream_combined_wrap] piece= 1 ... accepted= ...
  [sdf_stream_combined_wrap] seam_blend= cone_body overlap_samples= ...
  [sdf_stream_combined_wrap] negative= ... crop_evaluated= ...
  [sdf_stream_combined_wrap timing] total_s= ...

Compare to the measured 0045 r512 combined cone:
  evaluated=4,167,579; negative=386,655; crop=188^3,
  sign_changing_edges=73,656; active_cells/vertices=73,581;
  faces=147,312; total runtime=378.11 seconds.

Small SDF / DCSDD mesh changes can arise from fixed accepted-sample neural
batches instead of legacy np.array_split partitions; do not claim bitwise
identity without comparing the real mesh or SDF samples. The actual speedup
has not yet been measured on the full model/cluster.

TEST:
  PYTHONPATH=src:src/ngc:tests python -m unittest discover -s tests \
    -p 'test_sdf_combined_wrap_0046.py' -v
  PYTHONPATH=src:src/ngc:tests python -m unittest discover -s tests \
    -p 'test_sdf_zero_angular_wrap_0045.py' -v
  PYTHONPATH=src:src/ngc:tests python -m unittest discover -s tests \
    -p 'test_sdf_wrap_after_gate_0042.py' -v

Known limitations:
  * Only 0045-equivalent cone, not arbitrary combined wrap or puffer.
  * The missing-value mask behavior is preserved via sparse voxel ownership,
    but cone's support-adjacent cells must be rechecked at 512.
  * Other pipelines such as UDF cut and root-leaves are unchanged.
  * No implementation of independent rigging regions for a single fused mesh.

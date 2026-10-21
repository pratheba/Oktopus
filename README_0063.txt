Oktopus 0063 — DCSDD Python bridge profiling (no geometry changes)
===================================================================

Applies on top of the working 0062 code. The separate DCSDD C++ repo is
not included in the supplied Oktopus archive, so this patch profiles the
Python bridge only. The `pybind_cpp_contouring` stage is an opaque aggregate:
its native C++ internals (active cells, Hermite intersections, QEF solving,
face creation) CANNOT be distinguished without instrumenting that C++ repo.

Activate without modifying YAML or test_dc.py:

    OKTOPUS_DCSDD_PROFILE_0063=1 python test_dc.py \
      -c config/config_grid_b13d3_oktopus_puffer.yaml \
      -o test_adapt_oktopus_puffer_0063_body_profile_r512 \
      -s oktopus_9_v1 \
      -y adapt_oktopus_puffer_root_leaves_0062_body_bounded_profile.yaml \
      -r 512 -ckpt final -a sdf_dc \
      --dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd \
      --dcsdd-methods ours \
      --dcsdd-outer-iters 5 --dcsdd-inner-iters 3 \
      --dcsdd-mu 0.5 --dcsdd-dc-weight 0.4 --dcsdd-verbose \
      --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs \
      --dcsdd-crop-grid --dcsdd-fast-bbox-marking \
      --dcsdd-bounded-filtering --dcsdd-filter-batch-size 262144 \
      --sdf-localize-after-gate --surface-extraction ours

Extract just the DCSDD diagnostic stage lines and total runtime:

    grep -E '^\[dcsdd_profile_0063\]|^\[dcsdd\] timings|^time cost:' your_log_file.log

The same switch may be set using `dcsdd_profile_0063: true` in a config
dict that is already forwarded to the extractor. An explicit false overrides
the environment switch.

Stage notes:
 - grid_transfer_total_inclusive contains grid_native_validation,
   grid_crop and grid_coordinates (do not add them together).
 - safe_field_total_inclusive contains field_masks, field_clip,
   field_edt_and_roi and diagnostic printing.
 - total_run_inclusive includes conversion, diagnostics, field preparation,
   binding/C++ call, mesh construction, export, and mesh checks.
 - pybind_cpp_contouring covers any input conversion performed by the Python
   wrapper, native algorithm execution and result conversion; it is NOT a
   decomposition of the native C++ contourer.
 - Timings are wall-clock and include any contention; instrumentation is
   opt-in and must not be treated as a standalone speedup.

Once we see which Python bridge stage dominates, use native C++ source (if
needed) to instrument generating cells, active-edge enumeration, Hermite/QEF
assembly and solves, face construction, and pybind conversions separately.

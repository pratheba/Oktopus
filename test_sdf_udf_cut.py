"""Remove false SDF/MC caps using direct queries to the adapted UDF model.

The input MC mesh is already in world coordinates.  This runner reloads the
same adaptation YAML with the UDF model and evaluates that UDF directly at MC
triangle samples.  It never meshes or rescales the UDF.
"""

import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ("src", "src/app", "SDF", "UDF"):
    _full = _os.path.join(_ROOT, _p)
    if _full not in _sys.path:
        _sys.path.insert(0, _full)

import argparse
import os
import os.path as op
from time import time

import numpy as np
import torch
import yaml

from agent_3dvec_udf_cut import AgentUDFCut
from utils import DotDict, MCGrid, process_options


SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def start_test(opt):
    agent = AgentUDFCut()
    print("[test_sdf_udf_cut] agent = AgentUDFCut")

    config_path = op.join(opt.root_path, opt.config_path)
    output_path = op.join(
        opt.root_path,
        "inference",
        str(opt.num_samples),
        str(opt.out_path),
    )
    os.makedirs(output_path, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_path = opt.model_directory
    checkpoint = opt["checkpoints"][0]
    data_root = op.join(opt.root_path, config["dataset"]["root"])
    data_path = op.join(data_root, config["dataset"]["data_path"])

    t0 = time()
    agent.load_model(
        DEVICE,
        config_path,
        model_path,
        mode="train",
        checkpoint=checkpoint,
    )
    agent.load_data(data_root, data_path)
    print("UDF model and handle data loaded, time cost:", time() - t0)

    mc_grid = MCGrid(
        {
            "reso": int(opt.resolution),
            "level": 0.0,
            "size": float(opt.grid_size),
        }
    )

    manipulation_dir = op.join(opt.root_path, "exp", "train", "manipulation")
    adapt_file = (
        opt.test_file
        if op.isabs(opt.test_file)
        else op.join(manipulation_dir, opt.test_file)
    )
    if not op.isfile(adapt_file):
        raise FileNotFoundError(f"Adaptation YAML not found: {adapt_file}")

    shape_output = op.join(output_path, opt.shape_name)
    os.makedirs(shape_output, exist_ok=True)

    arg = {
        "exp_name": "adapt",
        "root_path": opt.root_path,
        "data_root": data_root,
        "data_path": data_path,
        "mc_grid": mc_grid,
        "output_folder": shape_output,
        "shape": opt.shape_name,
        "adapt_file": adapt_file,
        "checkpoint": checkpoint,
        "mc_mesh": opt.mc_mesh,
        "nsdudf_reference_mesh": opt.get("nsdudf_reference_mesh", None),
        "udf_far_value": opt.udf_far_value,
        "udf_model_batch_size": opt.udf_model_batch_size,
        "udf_cut_query_batch_size": opt.udf_cut_query_batch_size,
        "udf_cut_seed_world": opt.udf_cut_seed_world,
        "udf_cut_grow_world": opt.udf_cut_grow_world,
        "udf_cut_min_faces": opt.udf_cut_min_faces,
        "udf_cut_min_seed_faces": opt.udf_cut_min_seed_faces,
        "udf_cut_min_seed_fraction": opt.udf_cut_min_seed_fraction,
        "udf_cut_min_area_world2": opt.udf_cut_min_area_world2,
        "udf_cut_preserve_existing_boundaries": (
            not opt.udf_cut_allow_existing_boundary_touch
        ),
        "udf_cut_reference_min_edges": opt.udf_cut_reference_min_edges,
        "udf_cut_reference_min_span_world": (
            opt.udf_cut_reference_min_span_world
        ),
        "udf_cut_reference_max_distance_world": (
            opt.udf_cut_reference_max_distance_world
        ),
        "udf_cut_cleanup_boundary": opt.udf_cut_cleanup_boundary,
        "udf_cut_fill_small_holes": (
            not opt.udf_cut_no_fill_small_holes
        ),
        "udf_cut_fill_hole_max_edges": (
            opt.udf_cut_fill_hole_max_edges
        ),
        "udf_cut_fill_hole_max_perimeter_world": (
            opt.udf_cut_fill_hole_max_perimeter_world
        ),
        "udf_cut_fill_hole_max_span_world": (
            opt.udf_cut_fill_hole_max_span_world
        ),
        "udf_cut_boundary_smooth_iterations": (
            opt.udf_cut_boundary_smooth_iterations
        ),
        "udf_cut_boundary_smooth_lambda": (
            opt.udf_cut_boundary_smooth_lambda
        ),
        "udf_cut_boundary_smooth_mu": (
            opt.udf_cut_boundary_smooth_mu
        ),
        "udf_cut_boundary_smooth_min_edges": (
            opt.udf_cut_boundary_smooth_min_edges
        ),
        "udf_cut_boundary_max_step_fraction": (
            opt.udf_cut_boundary_max_step_fraction
        ),
        "udf_cut_boundary_max_total_fraction": (
            opt.udf_cut_boundary_max_total_fraction
        ),
        "udf_cut_boundary_redistribute": (
            opt.udf_cut_boundary_redistribute
        ),
        "udf_cut_boundary_redistribute_min_edges": (
            opt.udf_cut_boundary_redistribute_min_edges
        ),
        "udf_cut_boundary_redistribute_ring_count": (
            opt.udf_cut_boundary_redistribute_ring_count
        ),
        "udf_cut_boundary_curve_smooth_iterations": (
            opt.udf_cut_boundary_curve_smooth_iterations
        ),
        "udf_cut_boundary_curve_smooth_alpha": (
            opt.udf_cut_boundary_curve_smooth_alpha
        ),
        "udf_cut_boundary_harmonic_iterations": (
            opt.udf_cut_boundary_harmonic_iterations
        ),
        "udf_cut_boundary_strip_relax_iterations": (
            opt.udf_cut_boundary_strip_relax_iterations
        ),
        "udf_cut_boundary_strip_relax_step": (
            opt.udf_cut_boundary_strip_relax_step
        ),
        "udf_cut_boundary_redistribute_max_fraction": (
            opt.udf_cut_boundary_redistribute_max_fraction
        ),
        "udf_cut_boundary_strip_max_fraction": (
            opt.udf_cut_boundary_strip_max_fraction
        ),
        "udf_cut_boundary_min_area_ratio": (
            opt.udf_cut_boundary_min_area_ratio
        ),
        "udf_cut_boundary_min_normal_dot": (
            opt.udf_cut_boundary_min_normal_dot
        ),
    }

    print(
        "[test_sdf_udf_cut]",
        f"shape={opt.shape_name}",
        f"adapt_file={adapt_file}",
        f"mc_mesh={opt.mc_mesh}",
        f"seed={opt.udf_cut_seed_world}",
        f"grow={opt.udf_cut_grow_world}",
    )

    t0 = time()
    agent("part_adapt_udf_cut", arg)
    print("UDF cut time cost:", time() - t0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cut false SDF/MC caps using the aligned adapted UDF"
    )
    parser.add_argument("-ck", "--checkpoint_path", default="checkpoints")
    parser.add_argument(
        "-ckpt", "--checkpoints", type=str, nargs="+", default=["eval"]
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="UDF model config path.",
    )
    parser.add_argument("-o", "--out_path", required=True)
    parser.add_argument("-s", "--shape_name", required=True)
    parser.add_argument(
        "-y",
        "--test_file",
        required=True,
        help="The same adaptation YAML used for the SDF/MC run.",
    )
    parser.add_argument("-r", "--resolution", type=int, default=128)
    parser.add_argument("--grid-size", dest="grid_size", type=float, default=1.2)
    parser.add_argument(
        "--mc-mesh",
        dest="mc_mesh",
        required=True,
        help=(
            "World-space MC mesh. For multiple YAML adaptations, a format "
            "template may use {index}, {mode}, {accessory}, and {target}."
        ),
    )
    parser.add_argument(
        "--nsdudf-reference-mesh",
        dest="nsdudf_reference_mesh",
        default=None,
        help=(
            "Optional earlier open NSDUDF mesh. Only its large boundary "
            "components are used to gate candidate cap removal."
        ),
    )
    parser.add_argument("--udf-far-value", type=float, default=0.1)
    parser.add_argument("--udf-model-batch-size", type=int, default=32768)
    parser.add_argument("--udf-cut-query-batch-size", type=int, default=32768)
    parser.add_argument(
        "--udf-cut-seed-world",
        type=float,
        default=0.02,
        help="High-confidence cap seed threshold in world UDF units.",
    )
    parser.add_argument(
        "--udf-cut-grow-world",
        type=float,
        default=0.01,
        help="Lower threshold used to grow a connected cap patch.",
    )
    parser.add_argument("--udf-cut-min-faces", type=int, default=8)
    parser.add_argument("--udf-cut-min-seed-faces", type=int, default=2)
    parser.add_argument("--udf-cut-min-seed-fraction", type=float, default=0.05)
    parser.add_argument("--udf-cut-min-area-world2", type=float, default=0.0)
    parser.add_argument(
        "--udf-cut-allow-existing-boundary-touch",
        action="store_true",
        help=(
            "Allow deletion patches to touch boundaries that already exist "
            "in the input MC mesh. Disabled by default for safety."
        ),
    )
    parser.add_argument("--udf-cut-reference-min-edges", type=int, default=20)
    parser.add_argument(
        "--udf-cut-reference-min-span-world", type=float, default=0.05
    )
    parser.add_argument(
        "--udf-cut-reference-max-distance-world",
        type=float,
        default=None,
        help=(
            "When a reference mesh is supplied, reject a candidate patch if "
            "its boundary is farther than this from all retained reference "
            "boundary points."
        ),
    )
    parser.add_argument(
        "--udf-cut-cleanup-boundary",
        action="store_true",
        help=(
            "After cutting, fill only tiny closed holes and tangentially "
            "smooth the retained opening boundaries. Off by default."
        ),
    )
    parser.add_argument(
        "--udf-cut-no-fill-small-holes",
        action="store_true",
        help="Smooth boundaries but do not fill tiny closed loops.",
    )
    parser.add_argument(
        "--udf-cut-fill-hole-max-edges", type=int, default=24
    )
    parser.add_argument(
        "--udf-cut-fill-hole-max-perimeter-world", type=float, default=0.08
    )
    parser.add_argument(
        "--udf-cut-fill-hole-max-span-world", type=float, default=0.04
    )
    parser.add_argument(
        "--udf-cut-boundary-smooth-iterations", type=int, default=8
    )
    parser.add_argument(
        "--udf-cut-boundary-smooth-lambda", type=float, default=0.45
    )
    parser.add_argument(
        "--udf-cut-boundary-smooth-mu", type=float, default=-0.47
    )
    parser.add_argument(
        "--udf-cut-boundary-smooth-min-edges", type=int, default=12
    )
    parser.add_argument(
        "--udf-cut-boundary-max-step-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--udf-cut-boundary-max-total-fraction", type=float, default=0.75
    )
    parser.add_argument(
        "--udf-cut-boundary-redistribute",
        action="store_true",
        help=(
            "Uniformly redistribute each retained large cut boundary and "
            "relax only a narrow adjacent strip. Off by default."
        ),
    )
    parser.add_argument(
        "--udf-cut-boundary-redistribute-min-edges", type=int, default=12
    )
    parser.add_argument(
        "--udf-cut-boundary-redistribute-ring-count", type=int, default=1
    )
    parser.add_argument(
        "--udf-cut-boundary-curve-smooth-iterations", type=int, default=6
    )
    parser.add_argument(
        "--udf-cut-boundary-curve-smooth-alpha", type=float, default=0.45
    )
    parser.add_argument(
        "--udf-cut-boundary-harmonic-iterations", type=int, default=20
    )
    parser.add_argument(
        "--udf-cut-boundary-strip-relax-iterations", type=int, default=4
    )
    parser.add_argument(
        "--udf-cut-boundary-strip-relax-step", type=float, default=0.25
    )
    parser.add_argument(
        "--udf-cut-boundary-redistribute-max-fraction",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--udf-cut-boundary-strip-max-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--udf-cut-boundary-min-area-ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--udf-cut-boundary-min-normal-dot", type=float, default=0.0
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raw_opt = {
        "checkpoint_path": args.checkpoint_path,
        "root_path": op.dirname(op.abspath(__file__)),
        "checkpoints": args.checkpoints,
        "config_path": args.config_path,
        "out_path": args.out_path,
        "test_file": args.test_file,
        "shape_name": args.shape_name,
        "resolution": args.resolution,
        "grid_size": args.grid_size,
    }
    processed = process_options(raw_opt, mode="inference")
    opt = DotDict(processed)

    for key, value in vars(args).items():
        if key not in opt:
            opt[key] = value

    print(opt)
    start_test(opt)

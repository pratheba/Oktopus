"""Diagnose NSDUDF holes on an adapted Oktopus UDF.

This runner leaves the normal NSDUDF agent untouched. It uses
``AgentUDFNsdudfDiagnostic`` and calls:

    agent("part_adapt", arg)

Example:

    python test_nsdudf_diag.py \
        -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
        -o adapt_nsdudf_puffer \
        -s oktopus_9_v1 \
        -y test.yaml \
        -r 128 \
        --nsdudf-repo third_party/nsdudf \
        --nsdudf-grid 65 \
        --nsdudf-mesher marching_cubes \
        --nsdudf-oracle-chunk-size 16384 \
        --udf-batch-size 32768 \
        --nsdudf-diag-no-mesh
"""

# --- project path bootstrap ---
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ("src", "src/app", "SDF", "UDF"):
    _full = _os.path.join(_ROOT, _p)
    if _full not in _sys.path:
        _sys.path.insert(0, _full)
# --- end bootstrap ---

import argparse
import os
import os.path as op
from time import time

import numpy as np
import torch
import yaml

from agent_3dvec_udf_nsdudf_diag import AgentUDFNsdudfDiagnostic
from utils import DotDict, MCGrid, process_options


SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def start_test(opt):
    agent = AgentUDFNsdudfDiagnostic()
    print("[test_nsdudf_diag] agent = AgentUDFNsdudfDiagnostic")

    config_path = op.join(opt.root_path, opt.config_path)
    output_path = op.join(
        opt.root_path,
        "inference",
        str(opt.num_samples),
        str(opt.out_path),
    )
    os.makedirs(output_path, exist_ok=True)

    with open(config_path, "r") as handle:
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
    print("Model and handle data loaded, time cost:", time() - t0)

    grid_config = {
        "reso": int(opt.resolution),
        "level": 0.0,
        "size": float(opt.grid_size),
    }
    mc_grid = MCGrid(grid_config)

    manipulation_dir = op.join(
        opt.root_path,
        "exp",
        "train",
        "manipulation",
    )
    adapt_file = (
        opt.test_file
        if op.isabs(opt.test_file)
        else op.join(manipulation_dir, opt.test_file)
    )

    if not op.isfile(adapt_file):
        raise FileNotFoundError(
            f"Adaptation YAML not found: {adapt_file}"
        )

    shape_name = opt.shape_name
    shape_output = op.join(output_path, shape_name)
    os.makedirs(shape_output, exist_ok=True)
    if opt.get("nsdudf_diag_dir", None) is None:
        opt["nsdudf_diag_dir"] = op.join(shape_output, "nsdudf_diagnostics")

    arg = {
        "exp_name": "adapt",
        "data_root": data_root,
        "data_path": data_path,
        "mc_grid": mc_grid,
        "output_folder": shape_output,
        "shape": shape_name,
        "adapt_file": adapt_file,
        "checkpoint": checkpoint,
    }

    passthrough_keys = (
        "nsdudf_repo",
        "nsdudf_model",
        "nsdudf_grid",
        "nsdudf_normalize_udf",
        "nsdudf_oracle_chunk_size",
        "nsdudf_gradient_mode",
        "nsdudf_mesher",
        "nsdudf_dualmesh_batch_size",
        "udf_far_value",
        "udf_batch_size",
        "udf_domain_band",
        "udf_domain_padding",
        "udf_domain_scan_reso",
        "udf_cleanup",
        "nsdudf_min_component_faces",
        "nsdudf_merge_digits",
        "nsdudf_offset_world",
        "nsdudf_diag_dir",
        "nsdudf_diag_classifier_batch_size",
        "nsdudf_diag_query_slab",
        "nsdudf_diag_cell_slab",
        "nsdudf_diag_max_avg_factor",
        "nsdudf_diag_max_max_factor",
        "nsdudf_diag_loose_min_factor",
        "nsdudf_diag_max_points",
        "nsdudf_diag_extract_mesh",
    )

    for key in passthrough_keys:
        value = opt.get(key, None)
        if value is not None:
            arg[key] = value

    print(
        "[test_nsdudf]",
        f"shape={shape_name}",
        f"adapt_file={adapt_file}",
        f"grid_reso={opt.resolution}",
        f"nsdudf_grid={arg.get('nsdudf_grid')}",
        f"mesher={arg.get('nsdudf_mesher', 'marching_cubes')}",
        f"gradient_mode={arg.get('nsdudf_gradient_mode', 'finite_difference')}",
    )

    t0 = time()
    agent("part_adapt", arg)
    print("Adaptation time cost:", time() - t0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Diagnose NSDUDF holes on an adapted Oktopus UDF"
    )

    parser.add_argument(
        "-ck",
        "--checkpoint_path",
        default="checkpoints",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoints",
        type=str,
        nargs="+",
        default=["eval"],
    )
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-o", "--out_path", required=True)
    parser.add_argument("-s", "--shape_name", required=True)
    parser.add_argument(
        "-y",
        "--test_file",
        required=True,
        help=(
            "Adaptation YAML filename under exp/train/manipulation, "
            "or an absolute YAML path."
        ),
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=128,
        help="Oktopus support/prepass grid resolution.",
    )
    parser.add_argument(
        "--grid-size",
        dest="grid_size",
        type=float,
        default=1.2,
    )

    # NSDUDF
    parser.add_argument(
        "--nsdudf-repo",
        dest="nsdudf_repo",
        default=None,
    )
    parser.add_argument(
        "--nsdudf-model",
        dest="nsdudf_model",
        default=None,
    )
    parser.add_argument(
        "--nsdudf-grid",
        dest="nsdudf_grid",
        type=int,
        default=65,
        help="Samples per axis: 65 means 64 cells; 129 means 128 cells.",
    )
    parser.add_argument(
        "--nsdudf-oracle-chunk-size",
        dest="nsdudf_oracle_chunk_size",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--nsdudf-gradient-mode",
        dest="nsdudf_gradient_mode",
        choices=("finite_difference", "model_direct"),
        default="finite_difference",
    )
    parser.add_argument(
        "--nsdudf-mesher",
        dest="nsdudf_mesher",
        choices=("marching_cubes", "dual_mesh_udf"),
        default="marching_cubes",
    )
    parser.add_argument(
        "--nsdudf-dualmesh-batch-size",
        dest="nsdudf_dualmesh_batch_size",
        type=int,
        default=150000,
    )
    parser.add_argument(
        "--nsdudf-no-normalize",
        dest="nsdudf_no_normalize",
        action="store_true",
    )

    # Shared UDF domain/model-query controls
    parser.add_argument(
        "--udf-far-value",
        dest="udf_far_value",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--udf-batch-size",
        dest="udf_batch_size",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--udf-domain-band",
        dest="udf_domain_band",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--udf-domain-padding",
        dest="udf_domain_padding",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--udf-domain-scan-reso",
        dest="udf_domain_scan_reso",
        type=int,
        default=None,
    )

    # Conservative NSDUDF cleanup
    parser.add_argument(
        "--udf-cleanup",
        dest="udf_cleanup",
        action="store_true",
    )
    parser.add_argument(
        "--nsdudf-min-component-faces",
        dest="nsdudf_min_component_faces",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--nsdudf-merge-digits",
        dest="nsdudf_merge_digits",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--nsdudf-offset-world",
        dest="nsdudf_offset_world",
        type=float,
        default=0.0,
        help=(
            "Extract an offset UDF shell. Total approximate thickness "
            "is twice this value."
        ),
    )

    # Standalone diagnostics
    parser.add_argument(
        "--nsdudf-diag-dir",
        dest="nsdudf_diag_dir",
        default=None,
        help="Directory for summary.json, memmaps, and bad-face point clouds.",
    )
    parser.add_argument(
        "--nsdudf-diag-classifier-batch-size",
        dest="nsdudf_diag_classifier_batch_size",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--nsdudf-diag-query-slab",
        dest="nsdudf_diag_query_slab",
        type=int,
        default=4,
        help="Number of grid planes queried per diagnostic slab.",
    )
    parser.add_argument(
        "--nsdudf-diag-cell-slab",
        dest="nsdudf_diag_cell_slab",
        type=int,
        default=2,
        help="Number of cell planes classified per slab.",
    )
    parser.add_argument(
        "--nsdudf-diag-max-avg-factor",
        dest="nsdudf_diag_max_avg_factor",
        type=float,
        default=1.2,
        help="NSDUDF average-UDF threshold in voxel-size units.",
    )
    parser.add_argument(
        "--nsdudf-diag-max-max-factor",
        dest="nsdudf_diag_max_max_factor",
        type=float,
        default=2.0,
        help="NSDUDF maximum-corner-UDF threshold in voxel-size units.",
    )
    parser.add_argument(
        "--nsdudf-diag-loose-min-factor",
        dest="nsdudf_diag_loose_min_factor",
        type=float,
        default=1.0,
        help="Loose near-surface test used to count rejected surface cells.",
    )
    parser.add_argument(
        "--nsdudf-diag-max-points",
        dest="nsdudf_diag_max_points",
        type=int,
        default=200000,
        help="Maximum points written into each bad-face visualization PLY.",
    )
    parser.add_argument(
        "--nsdudf-diag-no-mesh",
        dest="nsdudf_diag_no_mesh",
        action="store_true",
        help="Run classification/consistency diagnostics without meshing.",
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

    forwarded = {
        "nsdudf_repo": args.nsdudf_repo,
        "nsdudf_model": args.nsdudf_model,
        "nsdudf_grid": args.nsdudf_grid,
        "nsdudf_normalize_udf": (
            False if args.nsdudf_no_normalize else True
        ),
        "nsdudf_oracle_chunk_size": args.nsdudf_oracle_chunk_size,
        "nsdudf_gradient_mode": args.nsdudf_gradient_mode,
        "nsdudf_mesher": args.nsdudf_mesher,
        "nsdudf_dualmesh_batch_size": args.nsdudf_dualmesh_batch_size,
        "udf_far_value": args.udf_far_value,
        "udf_batch_size": args.udf_batch_size,
        "udf_domain_band": args.udf_domain_band,
        "udf_domain_padding": args.udf_domain_padding,
        "udf_domain_scan_reso": args.udf_domain_scan_reso,
        "udf_cleanup": True if args.udf_cleanup else None,
        "nsdudf_min_component_faces": args.nsdudf_min_component_faces,
        "nsdudf_merge_digits": args.nsdudf_merge_digits,
        "nsdudf_offset_world": args.nsdudf_offset_world,
        "nsdudf_diag_dir": args.nsdudf_diag_dir,
        "nsdudf_diag_classifier_batch_size": args.nsdudf_diag_classifier_batch_size,
        "nsdudf_diag_query_slab": args.nsdudf_diag_query_slab,
        "nsdudf_diag_cell_slab": args.nsdudf_diag_cell_slab,
        "nsdudf_diag_max_avg_factor": args.nsdudf_diag_max_avg_factor,
        "nsdudf_diag_max_max_factor": args.nsdudf_diag_max_max_factor,
        "nsdudf_diag_loose_min_factor": args.nsdudf_diag_loose_min_factor,
        "nsdudf_diag_max_points": args.nsdudf_diag_max_points,
        "nsdudf_diag_extract_mesh": not args.nsdudf_diag_no_mesh,
    }

    for key, value in forwarded.items():
        if value is not None:
            opt[key] = value

    print(opt)
    start_test(opt)

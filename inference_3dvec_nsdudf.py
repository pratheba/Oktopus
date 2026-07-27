"""Runner: Oktopus UDF field  ->  NSDUDF pseudo-SDF meshing.

Mirrors ``inference_3dvec.py`` but drives ``AgentUDFNsdudf``, which reuses
the direct-query UDF oracle and meshes it with NSDUDF (neural pseudo-SDF +
custom marching cubes) instead of DualMesh-UDF.

No existing file is modified. Example:

    python inference_3dvec_nsdudf.py \
        -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
        -o puffer_nsdudf -s puffer -y test.txt -r 128 \
        --nsdudf-repo ~/Downloads/nsdudf-main --nsdudf-grid 129

Extract the uploaded nsdudf zip and build its custom_mc extension once
(see RECON_BENCH_README.md) before running.
"""

# --- project path bootstrap (same as inference_3dvec.py) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ('src', 'src/app', 'SDF', 'UDF'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
# --- end bootstrap ---

import os
import os.path as op
import argparse
from time import time

import numpy as np
import torch
import yaml

import app
from agent_3dvec_udf_nsdudf import AgentUDFNsdudf
from utils import MCGrid, process_options, DotDict


seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def start_inference(opt):
    agent = AgentUDFNsdudf()
    print('[inference_nsdudf] agent = AgentUDFNsdudf')

    config_path = op.join(opt.root_path, opt.config_path)
    output_path = op.join(opt.root_path, 'inference', str(opt.num_samples), str(opt.out_path))
    os.makedirs(output_path, exist_ok=True)

    config = yaml.safe_load(open(config_path)) if isinstance(config_path, str) else config_path

    model_path = opt.model_directory
    checkpoint = opt['checkpoints'][0]

    exp_name = 'train'
    data_root = op.join(opt.root_path, config['dataset']['root'])
    data_path = op.join(data_root, config['dataset']['data_path'])

    t0 = time()
    agent.load_model(device, config_path, model_path, mode='train', checkpoint=checkpoint)
    agent.load_data(data_root, data_path)
    print('Model and handle data loaded, time cost: ', time() - t0)

    grid_config = {'reso': opt.resolution, 'level': 0.0, 'size': 1.2}
    mc_grid = MCGrid(grid_config)
    arg = {
        'mc_grid': mc_grid,
        'exp_name': exp_name,
        'data_root': data_root,
        'shape': opt.shape_name,
        'output_folder': output_path,
        'checkpoint': checkpoint,
        'data_path': data_path,
    }

    for _k in (
        'nsdudf_repo',
        'nsdudf_model',
        'nsdudf_grid',
        'nsdudf_normalize_udf',
        'nsdudf_oracle_chunk_size',
        'udf_far_value',
        'udf_batch_size',
        'udf_domain_band',
        'udf_domain_padding',
        'udf_domain_scan_reso',
        'udf_cleanup',
    ):
    # Pass through NSDUDF + shared UDF-domain controls.
        _v = opt.get(_k, None) if hasattr(opt, 'get') else None
        if _v is not None:
            arg[_k] = _v

    agent('ngcnet_inference', arg)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Oktopus UDF -> NSDUDF pseudo-SDF meshing')
    p.add_argument('-ck', '--checkpoint_path', default='checkpoints')
    p.add_argument('-ckpt', '--checkpoints', type=str, nargs='+', default=['eval'])
    p.add_argument('-c', '--config_path', required=True)
    p.add_argument('-o', '--out_path', required=True)
    p.add_argument('-s', '--shape_name', required=True)
    p.add_argument('-y', '--test_file', required=True)
    p.add_argument('-r', '--resolution', type=int, default=128)
    # NSDUDF controls
    p.add_argument('--nsdudf-repo', dest='nsdudf_repo', default=None,
                   help="path to the extracted nsdudf checkout")
    p.add_argument('--nsdudf-model', dest='nsdudf_model', default=None,
                   help="path to the trained NSDUDF model.pt (default <repo>/model.pt)")
    p.add_argument('--nsdudf-grid', dest='nsdudf_grid', type=int, default=None,
                   help="pseudo-SDF grid samples/axis (129 or 257 for the dual-mesh variant)")
    p.add_argument(
        '--nsdudf-oracle-chunk-size',
        dest='nsdudf_oracle_chunk_size',
        type=int,
        default=32768,
        help=(
            "number of dense NSDUDF query points processed per finite-"
            "difference oracle chunk"
        ),
    )
    p.add_argument('--nsdudf-no-normalize', dest='nsdudf_no_normalize', action='store_true',
                   help="disable UDF/voxel normalization (only if the model was trained that way)")
    p.add_argument('--nsdudf-no-grads', dest='nsdudf_no_grads', action='store_true',
                   help="run the 8-input (no gradient) model variant")
    p.add_argument('--nsdudf-out7', dest='nsdudf_out7', action='store_true',
                   help="use the 7-output sigmoid model variant")
    # Shared UDF-domain controls (same meaning as inference_3dvec.py)
    p.add_argument('--udf-far-value', dest='udf_far_value', type=float, default=None)
    p.add_argument('--udf-batch-size', dest='udf_batch_size', type=int, default=None)
    p.add_argument('--udf-domain-band', dest='udf_domain_band', type=float, default=None)
    p.add_argument('--udf-domain-padding', dest='udf_domain_padding', type=float, default=None)
    p.add_argument('--udf-cleanup', dest='udf_cleanup', action='store_true')
    p.add_argument(
        "--nsdudf-gradient-mode",
        choices=("finite_difference", "model_direct"),
        default="finite_difference",
        help=(
            "NSDUDF gradient source. 'finite_difference' is currently "
            "implemented. 'model_direct' is reserved for the differentiable "
            "Oktopus-localization path."
        ),
    )
    p.add_argument(
        "--nsdudf-mesher",
        choices=("marching_cubes", "dual_mesh_udf"),
        default="marching_cubes",
        help=(
            "Final meshing backend after NSDUDF pseudo-sign prediction. "
            "'marching_cubes' is the current path; 'dual_mesh_udf' uses "
            "NSDUDF's relaxed DualMesh-UDF integration."
        ),
    )

    p.add_argument(
        "--nsdudf-dualmesh-batch-size",
        type=int,
        default=150000,
        help="Batch size for the NSDUDF + DualMesh-UDF backend.",
    )

    args = p.parse_args()
    opt = {
        'checkpoint_path': args.checkpoint_path,
        'root_path': op.dirname(op.abspath(__file__)),
        'checkpoints': args.checkpoints,
        'config_path': args.config_path,
        'out_path': args.out_path,
        'test_file': args.test_file,
        'shape_name': args.shape_name,
        'resolution': args.resolution,
    }
    opt = process_options(opt, mode='inference')
    opt = DotDict(opt)
    for _k, _v in [
        ('nsdudf_repo', args.nsdudf_repo),
        ('nsdudf_model', args.nsdudf_model),
        ('nsdudf_grid', args.nsdudf_grid),
        ('nsdudf_normalize_udf', (False if args.nsdudf_no_normalize else None)),
        ('nsdudf_use_grads', (False if args.nsdudf_no_grads else None)),
        ('nsdudf_gradient_mode', args.nsdudf_gradient_mode),
        ('nsdudf_out7', (True if args.nsdudf_out7 else None)),
        ('udf_far_value', args.udf_far_value),
        ('udf_batch_size', args.udf_batch_size),
        ('udf_domain_band', args.udf_domain_band),
        ('udf_domain_padding', args.udf_domain_padding),
        ('udf_cleanup', (True if args.udf_cleanup else None)),
        (
            'nsdudf_oracle_chunk_size',
            args.nsdudf_oracle_chunk_size,
        ),
        ('nsdudf_mesher', args.nsdudf_mesher),
        ('nsdudf_dualmesh_batch_size', args.nsdudf_dualmesh_batch_size),
    ]:
        if _v is not None:
            opt[_k] = _v
    print(opt)
    start_inference(opt)

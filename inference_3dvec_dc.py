"""Runner: Oktopus SDF field  ->  full run_ours.py reconstruction suite.

Mirrors ``inference_3dvec.py`` but drives ``AgentSDFDC``, whose surface
extractor converts the network SDF grid into the reference's
``(S, U, res)`` format and runs Marching Cubes, Dual Contouring, DC-SDD
("Ours"), Reach-for-the-Arcs and Kohlbrenner (cones / RC), saving each
mesh + timing.

No existing file is modified. Example:

    python inference_3dvec_dc.py \
        -c config/config_grid_b13d3_oktopus_dress1.yaml \
        -o dress1_recon -s dress1 -y test.txt -r 128 \
        --dcsdd-repo ~/Downloads/dual-contouring-of-signed-distance-data-main \
        --dcsdd-methods mc,dc,ours

Build the C++ bindings once (see RECON_BENCH_README.md) before running.
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

from agent_3dvec_sdf_dc import AgentSDFDC
from utils import MCGrid, process_options, DotDict


seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def start_inference(opt):
    agent = AgentSDFDC()
    print('[inference_dc] agent = AgentSDFDC')

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
        # DC-SDD benchmark controls:
        'surface_extraction': opt.get('surface_extraction', 'run_ours'),
        'level': opt.get('level', 0.0),
    }
    for _k in ('dcsdd_repo', 'dcsdd_methods', 'dcsdd_gt_mesh',
               'dcsdd_outer_iters', 'dcsdd_inner_iters', 'dcsdd_mu',
               'dcsdd_dc_weight', 'dcsdd_batch_size', 'dcsdd_verbose'):
        _v = opt.get(_k, None) if hasattr(opt, 'get') else None
        if _v is not None:
            arg[_k] = _v

    agent('ngcnet_inference', arg)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Oktopus SDF -> run_ours reconstruction suite')
    p.add_argument('-ck', '--checkpoint_path', default='checkpoints')
    p.add_argument('-ckpt', '--checkpoints', type=str, nargs='+', default=['eval'])
    p.add_argument('-c', '--config_path', required=True)
    p.add_argument('-o', '--out_path', required=True)
    p.add_argument('-s', '--shape_name', required=True)
    p.add_argument('-y', '--test_file', required=True)
    p.add_argument('-r', '--resolution', type=int, default=64)
    # DC-SDD suite controls
    p.add_argument('--surface-extraction', dest='surface_extraction', default='run_ours',
                   help="run_ours (all methods, default) or a single: mc|dc|ours|rfta|mnm1|mnm2")
    p.add_argument('--dcsdd-repo', dest='dcsdd_repo', default=None,
                   help="path to the dual-contouring-of-signed-distance-data checkout")
    p.add_argument('--dcsdd-methods', dest='dcsdd_methods', default=None,
                   help="comma list subset of mc,dc,ours,rfta,mnm1,mnm2")
    p.add_argument('--dcsdd-gt-mesh', dest='dcsdd_gt_mesh', default=None,
                   help="GT mesh (.obj/.ply) required only for the Kohlbrenner RC (mnm2) method")
    p.add_argument('--level', dest='level', type=float, default=0.0)
    p.add_argument('--dcsdd-outer-iters', dest='dcsdd_outer_iters', type=int, default=None)
    p.add_argument('--dcsdd-inner-iters', dest='dcsdd_inner_iters', type=int, default=None)
    p.add_argument('--dcsdd-mu', dest='dcsdd_mu', type=float, default=None)
    p.add_argument('--dcsdd-dc-weight', dest='dcsdd_dc_weight', type=float, default=None)
    p.add_argument('--dcsdd-batch-size', dest='dcsdd_batch_size', type=int, default=None)
    p.add_argument('--dcsdd-verbose', dest='dcsdd_verbose', action='store_true')
    p.add_argument(
        "--dcsdd-rfta-max-abs-sdf",
        type=float,
        default=0.05,
        help=(
            "Keep only RFTA samples with |S-level| at or below this "
            "metric distance. Removes Oktopus's saturated SDF plateau."
        ),
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
        ('surface_extraction', args.surface_extraction),
        ('dcsdd_repo', args.dcsdd_repo),
        ('dcsdd_methods', args.dcsdd_methods),
        ('dcsdd_gt_mesh', args.dcsdd_gt_mesh),
        ('level', args.level),
        ('dcsdd_outer_iters', args.dcsdd_outer_iters),
        ('dcsdd_inner_iters', args.dcsdd_inner_iters),
        ('dcsdd_mu', args.dcsdd_mu),
        ('dcsdd_dc_weight', args.dcsdd_dc_weight),
        ('dcsdd_batch_size', args.dcsdd_batch_size),
        ('dcsdd_verbose', args.dcsdd_verbose),
    ]:
        opt[_k] = _v
    print(opt)
    start_inference(opt)

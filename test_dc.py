# --- project path bootstrap (restructure: src/, SDF/, UDF/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ('src', 'src/app', 'SDF', 'UDF'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
# --- end bootstrap ---
from time import time
from app import AgentSDF, AgentUDF, AgentSDFasUDF
from app.agent_3dvec_sdf_dc import AgentSDFDC
import argparse
import os
import os.path as op

import torch
import yaml
from utils import MCGrid, process_options, DotDict

# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')
def start_test(opt):
    selected_agent = getattr(opt, 'agent', 'sdf_dc')
    _agent_cls = {
        'sdf_dc': AgentSDFDC,
        'sdf': AgentSDF,
        'udf': AgentUDF,
        'sdf_as_udf': AgentSDFasUDF,
    }.get(selected_agent, AgentSDFDC)
    agent = _agent_cls()
    print('[test] agent =', _agent_cls.__name__)

    config_path = op.join(opt.root_path, opt.config_path) # the config is yaml file path
    output_path = op.join(opt.root_path, 'inference', str(opt.num_samples), str(opt.out_path))
    os.makedirs(output_path, exist_ok=True)


    if isinstance(config_path, str):
        config =  yaml.safe_load(open(config_path))
    else:
        config = config_path

    model_path = opt.model_directory
    checkpoint = opt['checkpoints'][0]

    exp_name = 'train'
    #log_path = op.join(root_path, 'results', exp_name)
    data_root = op.join(opt.root_path, config['dataset']['root'])
    data_path = op.join(data_root, config['dataset']['data_path'])

    #output_path = op.join(root_path, 'inference', exp_name)
    os.makedirs(output_path, exist_ok=True)

    #agent.load_model(device, log_path, checkpoint='final')
    t0 = time()
    agent.load_model(device, config_path, model_path, mode='train', checkpoint=checkpoint)
    agent.load_data(data_root, data_path)
    print('Model and handle data loaded, time cost: ', time()-t0)

    # Marching Cubes config
    config_path = './exp/train/manipulation'
    grid_config = {
        'reso': args.resolution,
        'level': 0.,
        'size': 1.2,
    }

    t0 = time()
    mc_grid = MCGrid(grid_config)
#    shape_name = 'boots'
#
#    arg = {
#        'exp_name': 'adapt',
#        'data_root': data_root, 
#        'mc_grid': mc_grid,
#        'output_folder': op.join(output_path, f'{shape_name}'),
#        'shape': shape_name,
#        'adapt_file': op.join(config_path, f'adapt_{shape_name}.yaml'),
#    }
#    agent('part_adapt', arg)
#    print('time cost: ', time()-t0)
#############################################################################
    shape_name = args.shape_name #'oktopus_9_v1'
    test_file = args.test_file
    arg = {
        'exp_name': 'adapt',
        'data_root': data_root,
        'mc_grid': mc_grid,
        'output_folder': op.join(output_path, f'{shape_name}'),
        'shape': shape_name,
        'adapt_file': op.join(config_path, f'{test_file}'),
        'checkpoint': checkpoint,
        # AgentSDFDC surface extraction controls.
        'surface_extraction': args.surface_extraction,
        'level': args.level,
    }

    for _key in (
        'dcsdd_repo',
        'dcsdd_methods',
        'dcsdd_gt_mesh',
        'dcsdd_outer_iters',
        'dcsdd_inner_iters',
        'dcsdd_mu',
        'dcsdd_dc_weight',
        'dcsdd_batch_size',
        'dcsdd_sdf_band',
    ):
        _value = getattr(args, _key, None)
        if _value is not None:
            arg[_key] = _value

    if args.dcsdd_verbose:
        arg['dcsdd_verbose'] = True
    if getattr(args, 'extractor', None):
        arg['surface_extraction'] = args.extractor
    if getattr(args, 'udf_max_depth', None) is not None:
        arg['udf_max_depth'] = args.udf_max_depth
    if getattr(args, 'udf_domain_padding', None) is not None:
        arg['udf_domain_padding'] = args.udf_domain_padding
    if getattr(args, 'udf_domain_band', None) is not None:
        arg['udf_domain_band'] = args.udf_domain_band
    if getattr(args, 'udf_fd_cell_fraction', None) is not None:
        arg['udf_fd_cell_fraction'] = args.udf_fd_cell_fraction
    if getattr(args, 'udf_far_value', None) is not None:
        arg['udf_far_value'] = args.udf_far_value
    if getattr(args, 'udf_reliable', None) is not None:
        arg['udf_reliable_threshold'] = args.udf_reliable
    if getattr(args, 'udf_subdivide_threshold', None) is not None:
        arg['udf_subdivide_threshold'] = args.udf_subdivide_threshold
    if getattr(args, 'udf_projection_threshold', None) is not None:
        arg['udf_projection_threshold'] = args.udf_projection_threshold
    if getattr(args, 'udf_sample_threshold', None) is not None:
        arg['udf_sample_threshold'] = args.udf_sample_threshold
    if getattr(args, 'udf_sampling_depth', None) is not None:
        arg['udf_sampling_depth'] = args.udf_sampling_depth
    if getattr(args, 'udf_cleanup', False):
        arg['udf_cleanup'] = True
    if getattr(args, 'udf_no_fill_holes', False):
        arg['udf_fill_holes'] = False
    agent('part_adapt', arg)
    print('time cost: ', time()-t0)
################################################################################
#    shape_name = 'armadillo'
#    arg = {
#        'exp_name': 'mix',
#        'data_root': data_root, 
#        'mc_grid': mc_grid,
#        'output_folder': op.join(output_path, f'{shape_name}'),
#        'shape': shape_name,
#        'mixing_file': op.join(config_path, f'mix_{shape_name}.yaml'),
#    }
#    agent('part_mixing', arg)
#    print('time cost: ', time()-t0)
    
#    arg = {
#        'exp_name': 'mix',
#        'data_root': data_root, 
#        'mc_grid': mc_grid,
#        'output_folder': op.join(output_path, f'{shape_name}'),
#        'shape': shape_name,
#        'mixing_file': op.join(config_path, f'mix_{shape_name}.yaml'),
#    }
#    agent('part_mixing', arg)
#    print('time cost: ', time()-t0)

#    t0 = time()
#    mc_grid = MCGrid(grid_config)
#    shape_name = 'boots_2_v1'

#    arg = {
#        'exp_name': 'stretch',
#        'data_root': data_root, 
#        'mc_grid': mc_grid,
#        'output_folder': op.join(output_path, f'{shape_name}'),
#        'shape': shape_name,
#        'stretch_file': op.join(config_path, f'stretch_{shape_name}.yaml'),
#    }
#    agent('shape_stretch', arg)
#    print('time cost: ', time()-t0)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Oktopus adaptation with MC, DC and DC-SDD extraction')
    p.add_argument('-ck', '--checkpoint_path', required=False, default='checkpoints', help='Path to checkpoint directory.')
    p.add_argument('-ckpt', '--checkpoints', required=False, type=str, nargs="+", default=['eval'], help='checkpoints to evaluate.')
    p.add_argument('-c', '--config_path', required=True)
    p.add_argument('-o', '--out_path', required=True)
    p.add_argument('-s', '--shape_name', required=True)
    p.add_argument('-y', '--test_file', required=True)
    p.add_argument('-r', '--resolution', required=False, type=int, default=64)
    p.add_argument(
        '-a', '--agent',
        choices=['sdf_dc', 'sdf', 'udf', 'sdf_as_udf'],
        default='sdf_dc',
        help='agent to use; sdf_dc runs MC/DC/DC-SDD during adaptation',
    )
    p.add_argument(
        '--surface-extraction',
        dest='surface_extraction',
        default='run_ours',
        help='run_ours, mc, mc_ref, dc, ours, or rfta',
    )
    p.add_argument(
        '--dcsdd-repo',
        dest='dcsdd_repo',
        default=None,
        help='path to the built DC-SDD repository',
    )
    p.add_argument(
        '--dcsdd-methods',
        dest='dcsdd_methods',
        default='mc,mc_ref,dc,ours',
        help='comma-separated methods saved for every adapted SDF grid',
    )
    p.add_argument('--level', type=float, default=0.0)
    p.add_argument('--dcsdd-outer-iters', dest='dcsdd_outer_iters', type=int, default=None)
    p.add_argument('--dcsdd-inner-iters', dest='dcsdd_inner_iters', type=int, default=None)
    p.add_argument('--dcsdd-mu', dest='dcsdd_mu', type=float, default=None)
    p.add_argument('--dcsdd-dc-weight', dest='dcsdd_dc_weight', type=float, default=None)
    p.add_argument('--dcsdd-batch-size', dest='dcsdd_batch_size', type=int, default=None)
    p.add_argument('--dcsdd-sdf-band', dest='dcsdd_sdf_band', type=float, default=None)
    p.add_argument('--dcsdd-gt-mesh', dest='dcsdd_gt_mesh', default=None)
    p.add_argument('--dcsdd-verbose', dest='dcsdd_verbose', action='store_true')
    p.add_argument('-e', '--extractor', required=False, default=None,
                   help='legacy extractor override; normally leave unset in test_dc.py')
    p.add_argument('--udf-max-depth', dest='udf_max_depth', type=int, default=None,
                   help="DualMesh-UDF octree depth; 7 corresponds to about 128^3.")
    p.add_argument('--udf-domain-padding', dest='udf_domain_padding', type=float, default=None,
                   help="Padding around the low-UDF support bbox (default 0.15).")
    p.add_argument('--udf-domain-band', dest='udf_domain_band', type=float, default=None,
                   help="Raw model UDF band used only to choose the extraction bbox.")
    p.add_argument('--udf-fd-cell-fraction', dest='udf_fd_cell_fraction',
                   type=float, default=None,
                   help="Finite-difference step as a fraction of one octree cell.")
    p.add_argument('--udf-far-value', dest='udf_far_value', type=float, default=None,
                   help="Raw UDF returned outside adaptation support (default 0.1).")
    p.add_argument('--udf-reliable', dest='udf_reliable', type=float, default=None,
                   help="DualMesh reliable threshold in cube units (default 0.002).")
    p.add_argument('--udf-subdivide-threshold', dest='udf_subdivide_threshold',
                   type=float, default=None,
                   help="Octree adaptive-subdivide threshold (defaults to reliable).")
    p.add_argument('--udf-projection-threshold', dest='udf_projection_threshold',
                   type=float, default=None,
                   help="Grid-vertex validity threshold pu<thr (defaults to reliable).")
    p.add_argument('--udf-sample-threshold', dest='udf_sample_threshold',
                   type=float, default=None,
                   help="Dense-sample near-band threshold (default min(0.25*reliable,0.005)).")
    p.add_argument('--udf-sampling-depth', dest='udf_sampling_depth', type=int, default=None,
                   help="Dense-sampling octree depth.")
    p.add_argument('--udf-cleanup', dest='udf_cleanup', action='store_true',
                   help="Apply post-extraction cleanup (keep-largest, merge, fill holes, fix normals).")
    p.add_argument('--udf-no-fill-holes', dest='udf_no_fill_holes', action='store_true',
                   help="Disable hole-filling inside cleanup.")

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
            'extractor': args.extractor,
            'agent': args.agent,
            'surface_extraction': args.surface_extraction,
            'dcsdd_repo': args.dcsdd_repo,
            'dcsdd_methods': args.dcsdd_methods,
            'level': args.level,
    }

    opt = process_options(opt, mode='inference')
    print(opt)
    #opt['checkpoint_path'] = args.checkpoint_path
    #opt['root_path']= op.dirname(op.abspath(__file__))
    #opt['checkpoints']= args.checkpoints
    #opt['config_path']= args.config_path
    

    opt = DotDict(opt)
    
    print(opt)
    start_test(opt)

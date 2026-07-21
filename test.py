# --- project path bootstrap (restructure: src/, SDF/, UDF/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ('src', 'SDF', 'UDF'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
# --- end bootstrap ---
from time import time
from app import AgentSDF, AgentUDF, AgentSDFasUDF
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
    selected_agent = getattr(opt, 'agent', 'sdf')
    _agent_cls = {'udf': AgentUDF, 'sdf_as_udf': AgentSDFasUDF}.get(
        selected_agent, AgentSDF)
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
    }
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
    p = argparse.ArgumentParser(description='Input to test time config file')
    p.add_argument('-ck', '--checkpoint_path', required=False, default='checkpoints', help='Path to checkpoint directory.')
    p.add_argument('-ckpt', '--checkpoints', required=False, type=str, nargs="+", default=['eval'], help='checkpoints to evaluate.')
    p.add_argument('-c', '--config_path', required=True)
    p.add_argument('-o', '--out_path', required=True)
    p.add_argument('-s', '--shape_name', required=True)
    p.add_argument('-y', '--test_file', required=True)
    p.add_argument('-r', '--resolution', required=False, type=int, default=64)
    p.add_argument('-a', '--agent', choices=['sdf', 'udf', 'sdf_as_udf'],
                   default='sdf',
                   help="which agent: sdf (signed, default), udf (unsigned / "
                        "DualMeshUDF), or sdf_as_udf (diagnostic abs(SDF))")
    p.add_argument('-e', '--extractor', required=False, default=None,
                   help="SDF extractor override. AgentUDF always uses direct DualMesh-UDF.")
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
            'agent': args.agent
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

# --- project path bootstrap (restructure: src/, SDF/, UDF/) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
for _p in ('src', 'SDF', 'UDF'):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
# --- end bootstrap ---
from time import time
from app import Agent
import os, pickle, yaml, argparse
import os.path as op
import torch
import numpy as np
import yaml
from utils import MCGrid, process_options, DotDict


seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')



def start_inference1(opt):
    agent = Agent()

    config_path = op.join(opt.root_path, opt.config_path) # the config is yaml file path
    output_path = op.join(opt.root_path, 'inference', str(opt.num_samples), str(opt.out_path))
    os.makedirs(output_path, exist_ok=True)

    model_path = opt.model_directory
    for checkpoint in opt['checkpoints']:
        #print(checkpoint) 
        #checkpoint = 3000
        #agent.load_model(device, config_path, model_path, checkpoint=checkpoint)
        agent.load_model(device, config_path, model_path, mode='train', checkpoint=checkpoint)
        #print(model_path)
        #config_path = 'train'
        #agent.load_model(device,  model_path, config_path, checkpoint=checkpoint)
        mc_grid = MCGrid({
            'reso': 256,
            'level': 0.,
            'size': 1.2,
        })
        arg = {
            'mc_grid': mc_grid,
            'data_root': opt.data_root,
            'output_folder': output_path,
            'checkpoint': checkpoint,
            'data_path': opt.data_path,
        }
        agent('ngcnet_inference', arg)

def start_inference(opt):
    agent = Agent()

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
    t0 = time()

    # Marching Cubes config
    grid_config = {
        'reso': args.resolution,
        'level': 0.,
        'size': 1.2,
    }
    mc_grid = MCGrid(grid_config)
    arg = {
        'mc_grid': mc_grid,
        'exp_name': exp_name,
        'data_root': opt.data_root,
        'shape': opt.shape_name,
        'output_folder': output_path,
        'checkpoint': checkpoint,
        'data_path': data_path,
        'data_root': data_root,
    }
    agent('ngcnet_inference', arg)



#if __name__ == '__main__':
#    p = argparse.ArgumentParser(description='Input to inference config file')
#    p.add_argument('-ck', '--checkpoint_path', required=False, default='checkpoints', help='Path to checkpoint directory.')
#    p.add_argument('-ckpt', '--checkpoints', required=False, type=str, nargs="+", default=['eval'], help='checkpoints to evaluate.')
#    p.add_argument('-c', '--config_path', required=True)
#    p.add_argument('-o', '--out_path', required=True)
#    p.add_argument('-d', '--data_path', required=True)
#    p.add_argument('-r', '--data_root', required=True)
#
#    args = p.parse_args()
#    opt = {
#            'checkpoint_path': args.checkpoint_path,
#            'root_path': op.dirname(op.abspath(__file__)),
#            'checkpoints': args.checkpoints,
#            'config_path': args.config_path,
#            'out_path': args.out_path,
#            'data_path': args.data_path,
#            'data_root': args.data_root
#    }
#
#    opt = process_options(opt, mode='inference')
#    print(opt)
#    #opt['checkpoint_path'] = args.checkpoint_path
#    #opt['root_path']= op.dirname(op.abspath(__file__))
#    #opt['checkpoints']= args.checkpoints
#    #opt['config_path']= args.config_path
#    
#
#    opt = DotDict(opt)
#    
#    print(opt)
#    start_inference(opt)
#

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Input to inference time config file')
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
    start_inference(opt)

from time import time
from app import Agent
import os, pickle, yaml, argparse
import os.path as op
import torch
import numpy as np
import yaml
from utils import MCGrid, process_options, DotDict

# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')
def start_test(opt):
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

    # Marching Cubes config
    config_path = './exp/train/manipulation'
    grid_config = {
        'reso': 128,
        'level': 0.,
        'size': 1.0,
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

    args = p.parse_args()
    opt = {
            'checkpoint_path': args.checkpoint_path,
            'root_path': op.dirname(op.abspath(__file__)),
            'checkpoints': args.checkpoints,
            'config_path': args.config_path,
            'out_path': args.out_path,
            'test_file': args.test_file,
            'shape_name': args.shape_name
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

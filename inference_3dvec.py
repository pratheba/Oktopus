import os, pickle, yaml, argparse
import os.path as op
import torch
import numpy as np
from utils import MCGrid, process_options, DotDict
from app import Agent

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')



def start_inference(opt):
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
            'data_root': 'Pack10Dataset',
            'output_folder': output_path,
            'checkpoint': checkpoint,
        }
        agent('ngcnet_inference', arg)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Input to inference config file')
    p.add_argument('-ck', '--checkpoint_path', required=False, default='checkpoints', help='Path to checkpoint directory.')
    p.add_argument('-ckpt', '--checkpoints', required=False, type=str, nargs="+", default=['eval'], help='checkpoints to evaluate.')
    p.add_argument('-c', '--config_path', required=True)
    p.add_argument('-o', '--out_path', required=True)

    args = p.parse_args()
    opt = {
            'checkpoint_path': args.checkpoint_path,
            'root_path': op.dirname(op.abspath(__file__)),
            'checkpoints': args.checkpoints,
            'config_path': args.config_path,
            'out_path': args.out_path
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

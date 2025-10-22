import os, pickle, yaml
import os.path as op
import torch
import numpy as np
from utils import MCGrid
from app import Agent

seed = 2025
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

agent = Agent()

root_path = op.dirname(op.abspath(__file__))
exp_name = '4M'
config_path = op.join(root_path, 'results_resume', 'train') # exp_name)

additional_info = 'freq32_resume'
output_path = op.join(root_path, 'inference', additional_info,  exp_name)
os.makedirs(output_path, exist_ok=True)

model_path = ''
for checkpoint in ['63000']: #'3000', '6000','9000', 'final']:
    agent.load_model(device, config_path, model_path, checkpoint=checkpoint)#final')
    mc_grid = MCGrid({
        'reso': 512,
        'level': 0.,
    })
    arg = {
        'mc_grid': mc_grid,
        'data_root': 'Pack50Dataset',
        'output_folder': output_path,
        'checkpoint': checkpoint,
    }
    agent('ngcnet_inference', arg)

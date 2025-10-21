import os, argparse
import os.path as op
import numpy as np
import torch

import collections as _collections
import collections.abc as _abc
for _name in ("MutableMapping", "MutableSequence", "Mapping", "Sequence", "Set", "MutableSet"):
            if not hasattr(_collections, _name) and hasattr(_abc, _name):
                                    setattr(_collections, _name, getattr(_abc, _name))

import network, data, utils
from training import Trainer


### Set manual seed for debug
# seed = 2025
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = torch.device('cuda:0')

def start_trainer(args, opt):
    trainer = Trainer(opt, device)

    ### Train Dataset
    #train_dataloader = data.get_dataloader(opt['dataset'], dataset_mode='train')
    #opt['training']['train_dataloader'] = train_dataloader
    #val_dataloader = data.get_dataloader(opt['dataset'], dataset_mode='val')
    #opt['training']['val_dataloader'] = val_dataloader

    ### define model
    #if opt['training']['resume']:
    #    print("resuming")
    #    model, _ = utils.load_model(cpu_device, log_path, model_path, checkpoint)
    #else:
    #    model = network.define_model(opt['model'])
    #model.to(device)

    #opt['training']['train_loss'] = training.config_loss(opt['loss'])
    #opt['training']['val_loss'] = training.config_loss(opt['val_loss'])

    #trainmodel = Model(model, opt['training'])
    trainer.train_model() 

    if 'post_epochs' in opt['training']:
        trainer.optimize_code()

    # save the current config file to the output folder
    src_config_path = opt['config_path']
    cp_config_path = op.join(opt['log_path'], 'config.yaml')
    os.system(f'cp {src_config_path} {cp_config_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Input path to config file')
    p.add_argument('-c', '--config_path', required=True, help='Path to config file.')

    args = p.parse_args()
    opt = {
        'config_path': args.config_path,
        'root_path': op.dirname(op.abspath(__file__)),
    }
    opt = utils.process_options(opt, mode='train')
    start_trainer(args, opt)


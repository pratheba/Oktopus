'''Implements a generic training loop.
'''

import os, pickle
import os.path as op
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from dotted.collection import DottedDict
from time import time
from training.utils import get_optimizer, get_lr_scheduler, save_checkpoint
from loss import config_loss

import network, data


class Trainer:
    def __init__(self, opt, device='cuda:0'):
        self.model =None
        self.opt = opt
        self.device = device
        self.initialize()

    def initialize(self):
        ### Train Dataset
        self.train_dataloader = data.get_dataloader(self.opt['dataset'], dataset_mode='train')
        #opt['training']['train_dataloader'] = train_dataloader
        self.val_dataloader = data.get_dataloader(self.opt['dataset'], dataset_mode='val')
        #opt['training']['val_dataloader'] = val_dataloader

        ### define model
        if self.opt['training']['resume']:
            print("resuming")
            self.model, _ = utils.load_model(cpu_device, log_path, model_path, checkpoint)
        else:
            self.model = network.define_model(self.opt['model'])
        self.model.to(self.device)

        ### define loss
        self.train_loss_fn = config_loss(self.opt['loss']) 
        self.val_loss_fn = config_loss(self.opt['val_loss'])

    def train_model(self):
        opt = DottedDict(self.opt)
        res = get_optimizer(opt['training'], self.model)
        self.optim = res['optimizer']
        self.scheduler = res['epoch_lr']

        model_dir = opt.log_path

        os.makedirs(model_dir, exist_ok=True)
        summaries_dir = os.path.join(model_dir, 'summaries')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0
        best_val_epoch = -1
        best_train_epoch = -1
        best_val_loss = np.inf
        best_train_loss = np.inf

        train_dataloader = self.train_dataloader
        val_dataloader = self.val_dataloader

        opt = opt['training']

        #num_epochs = opt['training']['num_epochs']
        #print(self.model.model_dict())

        with tqdm(total=len(train_dataloader) * opt.num_epochs) as pbar:
            for epoch in range(opt.num_epochs):
                if not epoch % opt.epochs_til_ckpt and epoch:
                    #print(self.model.state_dict())
                    #exit()
                    save_checkpoint(self, best_train_epoch, best_train_loss, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

                # -----------------------
                # training
                # -----------------------
                epoch_train_loss = 0.0
                num_items = 0
                for data in train_dataloader:
                    model_input, gt, info = data
                    model_input = {key: val.cuda() for key,val in model_input.items()}
                    gt = {key: val.cuda() for key,val in gt.items()}
                    model_input['info'] = info
                    gt['info'] = info
                    batch_size = gt['sdf'].shape[0]
                    #print(model_input['samples_local'])
                    #exit()
                    model_output = self.model(model_input)
                    losses = self.train_loss_fn(model_output, gt)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        factor = opt.loss[loss_name].factor
                        loss_name = f'{loss_name}(X{factor})'
                        writer.add_scalar(loss_name, loss, total_steps)
                        train_loss += loss

                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    self.optim.zero_grad()
                    epoch_train_loss += (train_loss.item() * batch_size)
                    num_items += batch_size
                    train_loss.backward()

                    if opt.clip_grad:
                        if isinstance(opt.clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=opt.clip_grad)

                    self.optim.step()

                    pbar.update(1)
                    # default to be only one parameter group
                    current_lr = self.optim.param_groups[0]['lr']
                    writer.add_scalar('lr', current_lr, total_steps)

                    #message = "Epoch {}|Iter:{}, Loss {:0.4f}, Lr {:0.4f}".format(
                    #        epoch, total_steps, train_loss, current_lr)
                    #tqdm.write(message)
                    if best_train_loss >= epoch_train_loss:
                        best_train_loss = epoch_train_loss
                        best_train_epoch = epoch

                    if not total_steps % opt.steps_til_summary:
                        message = "Epoch {}|Iter:{}, Loss {:0.4f}, Lr {:0.4f}".format(
                            epoch, total_steps, epoch_train_loss, current_lr)
                        for name, loss in losses.items():
                            message = message + '{}(X{}): {:.4f}, '.format(name, opt.loss[name].factor, loss.item())
                        tqdm.write(message)


                    total_steps += 1
                    
                # -----------------------
                # Validation and epoch lr scheduler
                # -----------------------
                vals = {}
                if opt.val_type == 'None':
                    continue
                
                #print("val dataloader = ", val_dataloader, flush=True)
                epoch_val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    t1 =time()
                    # print('Validation Start')
                    #model.eval()
                    num_items = 0
                    for data in val_dataloader:
                        model_input, gt, info = data
                        model_input = {key: val.cuda() for key,val in model_input.items()}
                        gt = {key: val.cuda() for key,val in gt.items()}
                        model_input['info'] = info
                        batch_size = gt['sdf'].shape[0]

                        model_output = self.model.validation(model_input)
                        losses = self.val_loss_fn(model_output, gt)

                        val_loss = 0.
                        for loss_name, loss in losses.items():
                            factor = opt.loss[loss_name].factor
                            loss_name = f'{loss_name}(X{factor})'
                            writer.add_scalar(loss_name, loss, total_steps)
                            val_loss += loss

                        #writer.add_scalar("total_val_loss", val_loss, total_steps)

                        epoch_val_loss += (val_loss.item() * batch_size)
                        num_items += batch_size

                    epoch_val_loss /= num_items
                    if best_val_loss >= epoch_val_loss:
                        best_val_loss = epoch_val_loss
                        best_val_epoch = epoch
                        save_checkpoint(self, best_val_epoch, best_val_loss, os.path.join(checkpoints_dir, 'best_model_eval.pth'))

                    #scheduler.step(curr_epoch_loss)
                    if not total_steps % opt.steps_til_summary:
                        message = "Current Epoch val Loss {:0.4f}".format(epoch_val_loss) # epoch, total_steps, train_loss, current_lr)
                        tqdm.write(message)

                    self.scheduler.step(epoch_val_loss)
                self.model.train()

            # TODO:final evaluation
            save_checkpoint(self, epoch, epoch_train_loss, os.path.join(checkpoints_dir, 'model_final.pth'))
        

def optimize_code(opt, model):
    opt = DottedDict(opt)
    if hasattr(model, 'core'):
        embd = model.core.embd
    elif hasattr(model, 'encoder'):
        embd = model.encoder.embd
    else:
        embd = model.embd
    res = get_optimizer(opt, embd)
    optim = res['optimizer']

    model_dir = opt.log_path
    os.makedirs(model_dir, exist_ok=True)
    summaries_dir = os.path.join(model_dir, 'post_summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir)
    train_dataloader = opt['train_dataloader']
    #train_loss_fn = opt['train_loss']
    #val_loss_fn = opt['val_loss']

    total_steps = 0
    num_epochs = opt['post_epochs'] 
    model.set_post_mode()

    print('Start post optimization latent codes')
    with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
        for epoch in range(num_epochs):
            loss_recorder = []
            for data in train_dataloader:
                model_input, gt, info = data
                model_input = {key: val.cuda() for key,val in model_input.items()}
                gt = {key: val.cuda() for key,val in gt.items()}
                model_input['info'] = info
                gt['info'] = info

                model_output = model(model_input)
                losses = self.train_loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    train_loss += loss

                loss_recorder.append(train_loss.item())

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)
                total_steps += 1

            loss_mean = np.mean(loss_recorder)
            loss_max = np.max(loss_recorder)
            writer.add_scalar("Mean code post loss", loss_mean, epoch)
            writer.add_scalar("Max code post loss", loss_max, epoch)
            if epoch % 10 == 0:
                tqdm.write('Epoch{}| mean loss: {:.6e}, max loss: {:.6e}'.format(
                    epoch, loss_mean, loss_max
                ))

        torch.save(model.state_dict(), 
            os.path.join(checkpoints_dir, 'model_post.pth'))
        

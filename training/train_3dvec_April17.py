'''Implements a generic training loop for the 3-branch model
(base1 + base2 + detail) with 5 training phases.'''

import os, pickle
import os.path as op
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from dotted.collection import DottedDict
from time import time
from training.utils import (
    get_optimizer, get_lr_scheduler,
    save_checkpoint, load_model, load_checkpoint,
)
from loss_3dvec import config_loss
import network, data


class Trainer:
    def __init__(self, opt, device='cuda:0'):
        self.model = None
        self.opt = opt
        self.device = device
        self.initialize()

    def initialize(self):
        # Dataset
        self.train_dataloader = data.get_dataloader(
            self.opt['dataset'], dataset_mode='all')

        # Phase schedule
        ps = self.opt['training']['phase_schedule']
        self.E_b1 = int(ps['base1_epoch'])
        self.E_b2 = int(ps['base2_epoch'])
        self.E_bj = int(ps['base_epoch'])
        self.E_d  = int(ps['detail_epoch'])

        # Model
        self.model = network.define_model(
            self.opt['model'], self.opt['training']['phase_schedule'])
        if self.opt['training']['resume']:
            print("resuming")
            checkpoint_path = self.opt['logging_root']
            checkpoint = self.opt['training']['resume_data']['checkpoint']
            self.model = load_model(
                self.model, self.device, checkpoint_path, checkpoint)
        self.model.to(self.device)

        # Loss
        self.train_loss_fn = config_loss(self.opt['loss'])
        # self.val_loss_fn = config_loss(self.opt['val_loss'])

    def set_requires_grad(self, params, flag: bool):
        for p in params:
            p.requires_grad = flag

    def _set_phase_grads(self, epoch):
        """Toggle requires_grad per phase.

        Phases:
          P1 [0, E_b1)           : base1 only
          P2 [E_b1, E_b2)        : base2 only (base1 frozen)
          P3 [E_b2, E_bj)        : joint base1 + base2
          P4 [E_bj, E_d)         : detail only  (base frozen)
          P5 [E_d,  num_epochs)  : all joint
        """
        b1 = list(self.model.base_model.base1_parameters())
        b2 = list(self.model.base_model.base2_parameters())
        det = list(self.model.detail_model.detail_parameters())
        shared = list(self._shared_params())

        if epoch < self.E_b1:
            self.set_requires_grad(b1, True)
            self.set_requires_grad(b2, False)
            self.set_requires_grad(det, False)
            self.set_requires_grad(shared, True)
        elif epoch < self.E_b2:
            self.set_requires_grad(b1, False)
            self.set_requires_grad(b2, True)
            self.set_requires_grad(det, False)
            self.set_requires_grad(shared, True)
        elif epoch < self.E_bj:
            self.set_requires_grad(b1, True)
            self.set_requires_grad(b2, True)
            self.set_requires_grad(det, False)
            self.set_requires_grad(shared, True)
        elif epoch < self.E_d:
            self.set_requires_grad(b1, False)
            self.set_requires_grad(b2, False)
            self.set_requires_grad(det, True)
            self.set_requires_grad(shared, True)
        else:
            self.set_requires_grad(b1, True)
            self.set_requires_grad(b2, True)
            self.set_requires_grad(det, True)
            self.set_requires_grad(shared, True)

    def _shared_params(self):
        for m in self.model._shared_modules():
            for p in m.parameters():
                yield p

    def _step_optimizers(self, epoch):
        """Step the optimizer(s) relevant to the current phase."""
        if epoch < self.E_b1:
            self.optim_base1.step()
        elif epoch < self.E_b2:
            self.optim_base2.step()
        elif epoch < self.E_bj:
            self.optim_base1.step()
            self.optim_base2.step()
        elif epoch < self.E_d:
            self.optim_detail.step()
        else:
            self.optim_base1.step()
            self.optim_base2.step()
            self.optim_detail.step()

    def _zero_optimizers(self):
        self.optim_base1.zero_grad()
        self.optim_base2.zero_grad()
        self.optim_detail.zero_grad()

    def _step_schedulers(self, epoch, epoch_train_loss):
        if epoch < self.E_b1:
            self.scheduler_base1.step(epoch_train_loss)
        elif epoch < self.E_b2:
            self.scheduler_base2.step(epoch_train_loss)
        elif epoch < self.E_bj:
            self.scheduler_base1.step(epoch_train_loss)
            self.scheduler_base2.step(epoch_train_loss)
        elif epoch < self.E_d:
            self.scheduler_detail.step(epoch_train_loss)
        else:
            self.scheduler_base1.step(epoch_train_loss)
            self.scheduler_base2.step(epoch_train_loss)
            self.scheduler_detail.step(epoch_train_loss)

    def train_model(self):
        opt = DottedDict(self.opt)
        res = get_optimizer(opt['training'], self.model)
        # Expect `get_optimizer` to return optimizers & schedulers keyed per branch.
        self.optim_base1   = res['optimizer_base1']
        self.optim_base2   = res['optimizer_base2']
        self.optim_detail  = res['optimizer_detail']
        self.scheduler_base1  = res['epoch_lr_base1']
        self.scheduler_base2  = res['epoch_lr_base2']
        self.scheduler_detail = res['epoch_lr_detail']

        model_dir = opt.log_path
        os.makedirs(model_dir, exist_ok=True)
        summaries_dir = os.path.join(model_dir, 'summaries')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0
        best_train_epoch = -1
        best_train_loss = np.inf
        best_val_epoch = -1
        best_val_loss = np.inf

        train_dataloader = self.train_dataloader
        opt_train = opt['training']

        reset_epoch = 12000
        reset_count = 0
        print("number of iterations =", len(train_dataloader))

        with tqdm(total=len(train_dataloader) * opt_train.num_epochs) as pbar:
            for epoch in range(opt_train.num_epochs):
                # -- periodic checkpoint --
                if not epoch % opt_train.epochs_til_ckpt and epoch:
                    cur_b1 = self.optim_base1.param_groups[0]['lr']
                    cur_b2 = self.optim_base2.param_groups[0]['lr']
                    cur_d  = self.optim_detail.param_groups[0]['lr']
                    save_checkpoint(
                        self, best_train_epoch, best_train_loss,
                        {'base1': cur_b1, 'base2': cur_b2, 'detail': cur_d},
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch),
                    )

                # -- LR reset --
                if (epoch + 1) % reset_epoch == 0:
                    reset_count += 1
                    self.optim_base1.param_groups[0]['lr']  = self.opt['training']['optim_base1']['lr']  * (0.5 / reset_count)
                    self.optim_base2.param_groups[0]['lr']  = self.opt['training']['optim_base2']['lr']  * (0.5 / reset_count)
                    self.optim_detail.param_groups[0]['lr'] = self.opt['training']['optim_detail']['lr'] * (0.5 / reset_count)

                # -- set grad flags for this phase --
                self._set_phase_grads(epoch)

                # -- one training epoch --
                epoch_train_loss = 0.0
                num_items = 0
                for batch in train_dataloader:
                    model_input, gt, info = batch
                    model_input = {k: v.cuda() for k, v in model_input.items()}
                    gt = {k: v.cuda() for k, v in gt.items()}
                    model_input['info'] = info
                    gt['info'] = info

                    batch_size = gt['sdf'].shape[0]

                    model_output = self.model(model_input, epoch)
                    losses = self.train_loss_fn(
                        model_output, gt,
                        epoch=epoch, E0=self.E_b1, E1=self.E_b2, E2=self.E_bj, E3=self.E_d,
                    )

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        factor = opt_train.loss[loss_name].factor
                        log_name = f'{loss_name}(X{factor})'
                        writer.add_scalar(log_name, loss, total_steps)
                        train_loss = train_loss + loss

                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    self._zero_optimizers()
                    epoch_train_loss += (train_loss.item() * batch_size)
                    num_items += batch_size
                    train_loss.backward()

                    if opt_train.clip_grad:
                        max_norm = (1. if isinstance(opt_train.clip_grad, bool)
                                    else opt_train.clip_grad)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=max_norm)

                    self._step_optimizers(epoch)
                    pbar.update(1)
                    total_steps += 1
                    total_steps += 1

                cur_b1 = self.optim_base1.param_groups[0]['lr']
                cur_b2 = self.optim_base2.param_groups[0]['lr']
                cur_d  = self.optim_detail.param_groups[0]['lr']
                writer.add_scalar('base1_lr',  cur_b1, total_steps)
                writer.add_scalar('base2_lr',  cur_b2, total_steps)
                writer.add_scalar('detail_lr', cur_d,  total_steps)

                epoch_train_loss /= max(1, num_items)
                if best_train_loss >= epoch_train_loss:
                    best_train_loss = epoch_train_loss
                    best_train_epoch = epoch
                    save_checkpoint(
                        self, best_train_epoch, best_train_loss,
                        {'base1': cur_b1, 'base2': cur_b2, 'detail': cur_d},
                        os.path.join(checkpoints_dir, 'best_model_train.pth'),
                    )

                if not total_steps % opt_train.steps_til_summary:
                    msg = (f"Epoch train {epoch}|Iter:{total_steps}, "
                           f"Loss {epoch_train_loss:.4f}, "
                           f"B1_Lr {cur_b1:.8f}, B2_Lr {cur_b2:.8f}, D_Lr {cur_d:.8f}\n")
                    for name, loss in losses.items():
                        msg += '{}(X{}): {:.4f}, \n'.format(
                            name, opt_train.loss[name].factor, loss.item())
                    tqdm.write(msg)


                # -- validation / scheduler step --
                if opt_train.val_type == 'None':
                    self._step_schedulers(epoch, epoch_train_loss)
                    continue
                # NOTE: validation code below is unchanged in behavior
                # (it would need a val dataloader). Left as placeholder.

            # final save
            save_checkpoint(
                self, epoch, epoch_train_loss,
                {'base1': cur_b1, 'base2': cur_b2, 'detail': cur_d},
                os.path.join(checkpoints_dir, 'model_final.pth'),
            )


def optimize_code(opt, model):
    opt = DottedDict(opt)
    if hasattr(model, 'core'):
        embd = model.core.embd
    elif hasattr(model, 'encoder'):
        embd = model.encoder.embd
    else:
        embd = model.curve_embd if hasattr(model, 'curve_embd') else model.embd
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

    total_steps = 0
    num_epochs = opt['post_epochs']
    model.set_post_mode()

    print('Start post optimization latent codes')
    with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
        for epoch in range(num_epochs):
            loss_recorder = []
            for batch in train_dataloader:
                model_input, gt, info = batch
                model_input = {k: v.cuda() for k, v in model_input.items()}
                gt = {k: v.cuda() for k, v in gt.items()}
                model_input['info'] = info
                gt['info'] = info

                model_output = model(model_input)
                # NOTE: original code referenced self.train_loss_fn here;
                # we expect `opt['train_loss_fn']` to be passed in instead.
                losses = opt['train_loss_fn'](model_output, gt)

                train_loss = 0.
                for _, loss in losses.items():
                    train_loss = train_loss + loss
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
                    epoch, loss_mean, loss_max))

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_post.pth'))

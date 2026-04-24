
"""Training loop with automatic epoch-driven internal modes.

Internal stages:
  - base_train                 : [0, base_epoch)
      - base1 only            : [0, base1_epoch)
      - base2 only            : [base1_epoch, base2_epoch)
      - base joint            : [base2_epoch, base_epoch)
  - detail_train_from_base_ckpt: [base_epoch, detail_epoch)
      base/shared frozen, detail-only optimizer/scheduler reset at boundary
  - joint_finetune            : [detail_epoch, final_epoch)
      all trainable, optimizers/schedulers reset again at boundary

The user does NOT set mode manually. Mode is inferred from epoch.
"""

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from dotted.collection import DottedDict

from training.utils import get_optimizer, save_checkpoint, load_model, load_checkpoint
from loss_3dvec import config_loss
import network, data


class Trainer:
    def __init__(self, opt, device='cuda:0'):
        self.model = None
        self.opt = opt
        self.device = device
        self.initialize()

    def initialize(self):
        self.train_dataloader = data.get_dataloader(
            self.opt['dataset'], dataset_mode='all'
        )

        ps = self.opt['training']['phase_schedule']
        self.E_b1 = int(ps['base1_epoch'])
        self.E_b2 = int(ps['base2_epoch'])
        self.E_bj = int(ps['base_epoch'])
        self.E_d = int(ps['detail_epoch'])
        self.E_f = int(ps['final_epoch'])

        self.model = network.define_model(
            self.opt['model'], self.opt['training']['phase_schedule']
        )

        #if self.opt['training'].get('resume', False):
        #    checkpoint_path = self.opt['logging_root']
        #    checkpoint = self.opt['training']['resume_data']['checkpoint']
        #    self.model = load_model(self.model, self.device, checkpoint_path, checkpoint)

        self.model.to(self.device)
        self.train_loss_fn = config_loss(self.opt['loss'])

        self._current_stage_name = None
        self._best_stage_loss = np.inf
        self._best_stage_epoch = -1

    def set_requires_grad(self, params, flag: bool):
        for p in params:
            p.requires_grad = flag

    def _shared_params(self):
        for m in self.model._shared_modules():
            for p in m.parameters():
                yield p

    def _branch_params(self):
        return {
            'b1': list(self.model.base_model.base1_parameters()),
            'b2': list(self.model.base_model.base2_parameters()),
            'det': list(self.model.detail_model.detail_parameters()),
            'shared': list(self._shared_params()),
        }

    def _epoch_mode(self, epoch):
        if epoch < self.E_bj:
            return 'base_train'
        if epoch < self.E_d:
            return 'detail_train_from_base_ckpt'
        return 'joint_finetune'

    def _phase_name(self, epoch):
        if epoch < self.E_b1:
            return 'base1_only'
        if epoch < self.E_b2:
            return 'base2_only'
        if epoch < self.E_bj:
            return 'base_joint'
        if epoch < self.E_d:
            return 'detail_only'
        return 'all_joint'

    def _effective_epoch(self, epoch):
        # Force model's internal phase selection consistently.
        if epoch < self.E_bj:
            return epoch
        if epoch < self.E_d:
            return self.E_bj   # detail_only branch inside model
        return self.E_d        # all_joint branch inside model

    def _set_phase_grads(self, epoch):
        P = self._branch_params()
        mode = self._epoch_mode(epoch)

        if mode == 'detail_train_from_base_ckpt':
            self.set_requires_grad(P['b1'], False)
            self.set_requires_grad(P['b2'], False)
            self.set_requires_grad(P['shared'], False)
            self.set_requires_grad(P['det'], True)
            return

        if mode == 'joint_finetune':
            self.set_requires_grad(P['b1'], True)
            self.set_requires_grad(P['b2'], True)
            self.set_requires_grad(P['shared'], True)
            self.set_requires_grad(P['det'], True)
            return

        # base_train sub-phases
        if epoch < self.E_b1:
            self.set_requires_grad(P['b1'], True)
            self.set_requires_grad(P['b2'], False)
            self.set_requires_grad(P['shared'], True)
            self.set_requires_grad(P['det'], False)
        elif epoch < self.E_b2:
            self.set_requires_grad(P['b1'], False)
            self.set_requires_grad(P['b2'], True)
            self.set_requires_grad(P['shared'], True)
            self.set_requires_grad(P['det'], False)
        else:
            self.set_requires_grad(P['b1'], True)
            self.set_requires_grad(P['b2'], True)
            self.set_requires_grad(P['shared'], True)
            self.set_requires_grad(P['det'], False)

    def _build_optimizers_and_schedulers(self):
        opt = DottedDict(self.opt)
        res = get_optimizer(opt['training'], self.model)
        self.optim_base1 = res['optimizer_base1']
        self.optim_base2 = res['optimizer_base2']
        self.optim_detail = res['optimizer_detail']
        self.scheduler_base1 = res['epoch_lr_base1']
        self.scheduler_base2 = res['epoch_lr_base2']
        self.scheduler_detail = res['epoch_lr_detail']

    def _reset_for_new_stage(self, epoch, checkpoints_dir):
        stage = self._epoch_mode(epoch)
        self._build_optimizers_and_schedulers()
        self._best_stage_loss = np.inf
        self._best_stage_epoch = -1
        self._current_stage_name = stage

        # Save a boundary checkpoint right when the stage starts.
        boundary_name = None
        if epoch == self.E_bj:
            boundary_name = 'base_end.pth'
        elif epoch == self.E_d:
            boundary_name = 'detail_end.pth'

        if boundary_name is not None:
            save_checkpoint(
                self,
                epoch=epoch,
                loss=np.inf,
                lrs=self._current_lrs(),
                filename=os.path.join(checkpoints_dir, boundary_name),
            )

    def _reset_detail_stage_only(self):
        """
        Start detail stage fresh, but keep trained base optimizer/scheduler state.
        Only detail optimizer/scheduler are recreated.
        """
        opt = DottedDict(self.opt)
        res = get_optimizer(opt['training'], self.model)

        self.optim_detail = res['optimizer_detail']
        self.scheduler_detail = res['epoch_lr_detail']

    def _reset_rop_scheduler_state(self, sch):
        if sch is None:
            return
        if sch.__class__.__name__ == 'ReduceLROnPlateau':
            sch.best = float('inf')
            sch.num_bad_epochs = 0
            sch.cooldown_counter = 0

    def _reset_stage_tracking(self, stage_name):
        self._best_stage_loss = np.inf
        self._best_stage_epoch = -1
        self._current_stage_name = stage_name

    def _zero_optimizers(self):
        self.optim_base1.zero_grad()
        self.optim_base2.zero_grad()
        self.optim_detail.zero_grad()

    def _step_optimizers(self, epoch):
        mode = self._epoch_mode(epoch)
        if mode == 'detail_train_from_base_ckpt':
            self.optim_detail.step()
            return
        if mode == 'joint_finetune':
            self.optim_base1.step()
            self.optim_base2.step()
            self.optim_detail.step()
            return

        # base_train
        if epoch < self.E_b1:
            self.optim_base1.step()
        elif epoch < self.E_b2:
            self.optim_base2.step()
        else:
            self.optim_base1.step()
            self.optim_base2.step()

    def _scheduler_step(self, sch, val):
        if sch is None:
            return
        if sch.__class__.__name__ == 'ReduceLROnPlateau':
            sch.step(val)
        else:
            sch.step()

    def _step_schedulers(self, epoch, epoch_train_loss):
        mode = self._epoch_mode(epoch)
        if mode == 'detail_train_from_base_ckpt':
            self._scheduler_step(self.scheduler_detail, epoch_train_loss)
            return
        if mode == 'joint_finetune':
            self._scheduler_step(self.scheduler_base1, epoch_train_loss)
            self._scheduler_step(self.scheduler_base2, epoch_train_loss)
            self._scheduler_step(self.scheduler_detail, epoch_train_loss)
            return

        if epoch < self.E_b1:
            self._scheduler_step(self.scheduler_base1, epoch_train_loss)
        elif epoch < self.E_b2:
            self._scheduler_step(self.scheduler_base2, epoch_train_loss)
        else:
            self._scheduler_step(self.scheduler_base1, epoch_train_loss)
            self._scheduler_step(self.scheduler_base2, epoch_train_loss)

    def _load_lrs_from_checkpoint(self, checkpoint_path, checkpoint_name):
        ckpt_path = os.path.join(checkpoint_path, 'train', 'checkpoints', checkpoint_name)
        ckpt = torch.load(ckpt_path, map_location='cpu')

        lr_base1 = ckpt.get('lr_base1', None)
        lr_base2 = ckpt.get('lr_base2', None)
        lr_detail = ckpt.get('lr_detail', None)

        if lr_base1 is not None:
            for g in self.optim_base1.param_groups:
                g['lr'] = lr_base1

        if lr_base2 is not None:
            for g in self.optim_base2.param_groups:
                g['lr'] = lr_base2

        if lr_detail is not None:
            for g in self.optim_detail.param_groups:
                g['lr'] = lr_detail

    def _current_lrs(self):
        return {
            'base1': self.optim_base1.param_groups[0]['lr'],
            'base2': self.optim_base2.param_groups[0]['lr'],
            'detail': self.optim_detail.param_groups[0]['lr'],
        }

    def _boundary_checkpoint_name(self, epoch):
        # Save when finishing the previous phase (end of this epoch).
        if epoch + 1 == self.E_b1:
            return 'base1_end.pth'
        if epoch + 1 == self.E_b2:
            return 'base2_end.pth'
        if epoch + 1 == self.E_bj:
            return 'base_end.pth'
        if epoch + 1 == self.E_d:
            return 'detail_end.pth'
        if epoch + 1 == self.E_f:
            return 'final_end.pth'
        return None

    def train_model(self):
        opt = DottedDict(self.opt)
        model_dir = opt.log_path
        os.makedirs(model_dir, exist_ok=True)
        summaries_dir = os.path.join(model_dir, 'summaries')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        writer = SummaryWriter(summaries_dir)

        self._build_optimizers_and_schedulers()
        self._current_stage_name = self._epoch_mode(0)

        total_steps = 0
        best_train_loss = np.inf
        best_train_epoch = -1
        train_dataloader = self.train_dataloader
        opt_train = opt['training']
        start_epoch = 0
        if opt_train.get('resume', False):
            rd = opt_train['resume_data']
            start_epoch = int(rd.get('start_epoch', 0))

            if opt_train['resume']:
                checkpoint_path = opt_train['resume_data'].get('checkpoint_path', self.opt['logging_root'])
            else:
                checkpoint_path = self.opt['logging_root']
            checkpoint = rd['checkpoint']

            if rd.get('load_model', True):
                self.model = load_model(
                    self.model,
                    self.device,
                    checkpoint_path,
                    checkpoint,
                )

            if any([
                rd.get('load_optimizer_base1', False),
                rd.get('load_optimizer_base2', False),
                rd.get('load_optimizer_detail', False),
                rd.get('load_scheduler_base1', False),
                rd.get('load_scheduler_base2', False),
                rd.get('load_scheduler_detail', False),
            ]):
                load_checkpoint(
                    self.model,
                    checkpoint_path,
                    self.device,
                    checkpoint=checkpoint,
                    optim_base1=self.optim_base1 if rd.get('load_optimizer_base1', False) else None,
                    optim_base2=self.optim_base2 if rd.get('load_optimizer_base2', False) else None,
                    optim_detail=self.optim_detail if rd.get('load_optimizer_detail', False) else None,
                    scheduler_base1=self.scheduler_base1 if rd.get('load_scheduler_base1', False) else None,
                    scheduler_base2=self.scheduler_base2 if rd.get('load_scheduler_base2', False) else None,
                    scheduler_detail=self.scheduler_detail if rd.get('load_scheduler_detail', False) else None,
                    strict=False,
                )


        if start_epoch < self.E_b2:
            self._current_stage_name = 'base2_only' if start_epoch >= self.E_b1 else 'base1_only'
        elif start_epoch < self.E_bj:
            self._current_stage_name = 'base_joint'
        elif start_epoch < self.E_d:
            self._current_stage_name = 'detail_only'
        else:
            self._current_stage_name = 'final_joint'

        self._best_stage_loss = np.inf
        self._best_stage_epoch = -1

        print('number of iterations =', len(train_dataloader))
        self.model.train()

        with tqdm(total=len(train_dataloader) * opt_train.num_epochs) as pbar:
            for epoch in range(start_epoch, opt_train.num_epochs):
                # Automatic stage reset exactly at boundaries.
                # Stage boundary bookkeeping only. No optimizer/scheduler reset.
                if epoch == self.E_b2:
                    # entering base_joint
                    self._reset_stage_tracking('base_joint')
                    self._reset_rop_scheduler_state(self.scheduler_base1)
                    self._reset_rop_scheduler_state(self.scheduler_base2)

                elif epoch == self.E_bj:
                    # entering detail_only
                    self._reset_stage_tracking('detail_only')
                    self._reset_rop_scheduler_state(self.scheduler_detail)

                elif epoch == self.E_d:
                    # entering final all-joint
                    self._reset_stage_tracking('final_joint')
                    self._reset_rop_scheduler_state(self.scheduler_base1)
                    self._reset_rop_scheduler_state(self.scheduler_base2)
                    self._reset_rop_scheduler_state(self.scheduler_detail)

#                if epoch == self.E_bj or epoch == self.E_d:
#                    self._set_phase_grads(epoch)
#                    self._reset_for_new_stage(epoch, checkpoints_dir)
                # Stage boundary handling
#                if epoch == self.E_bj:
#                    # entering detail-only stage:
#                    # keep trained base optimizer/scheduler state,
#                    # reset only detail optimizer/scheduler
#                    self._set_phase_grads(epoch)
#                    self._reset_detail_stage_only()
#
#                    self._best_stage_loss = np.inf
#                    self._best_stage_epoch = -1
#                    self._current_stage_name = self._epoch_mode(epoch)
#
#                elif epoch == self.E_d:
#                    # entering final joint stage:
#                    # do NOT reset base optimizer/scheduler
#                    # just switch stage bookkeeping
#                    self._set_phase_grads(epoch)
#
#                    self._best_stage_loss = np.inf
#                    self._best_stage_epoch = -1
#                    self._current_stage_name = self._epoch_mode(epoch)

                if not epoch % opt_train.epochs_til_ckpt and epoch:
                    save_checkpoint(
                        self, best_train_epoch, best_train_loss, self._current_lrs(),
                        os.path.join(checkpoints_dir, f'epoch_{epoch:04d}.pth'),
                    )

                self._set_phase_grads(epoch)
                eff_epoch = self._effective_epoch(epoch)
                mode = self._epoch_mode(epoch)
                phase = self._phase_name(epoch)

                epoch_train_loss = 0.0
                num_items = 0

                for batch in train_dataloader:
                    model_input, gt, info = batch
                    model_input = {k: v.cuda() for k, v in model_input.items()}
                    gt = {k: v.cuda() for k, v in gt.items()}
                    model_input['info'] = info
                    gt['info'] = info

                    batch_size = gt['sdf'].shape[0]
                    #model_output = self.model(model_input, eff_epoch)
                    model_output = self.model(model_input, epoch)

                    #if phase == 'detail_only':
#                        print("epoch:", epoch)
#                        print("eff_epoch:", eff_epoch)
#                        print("phase_out:", model_output['phase'])
#                        print("alpha_detail:", model_output['alpha_detail'])
#                        print("gate_detail_mean:", model_output['gate_detail'].mean().item())
#                        print("sdf_detail_abs_mean:", model_output['sdf_detail'].abs().mean().item())
#                        print("pred_detail_res_abs_mean:",
#                              (model_output['gate_detail'] * model_output['sdf_detail']).abs().mean().item())
#                        print("final_minus_base_abs_mean:",
#                              (model_output['sdf'] - model_output['sdf_base']).abs().mean().item(), flush=True)

                    losses = self.train_loss_fn(
                        model_output, gt,
                        epoch=epoch,
                        E0=self.E_b1, E1=self.E_b2, E2=self.E_bj, E3=self.E_d,
                        mode=mode,
                    )

                    train_loss = 0.0
                    for loss_name, loss in losses.items():
                        factor = opt_train.loss[loss_name].factor
                        writer.add_scalar(f'{loss_name}(X{factor})', loss, total_steps)
                        train_loss = train_loss + loss

                    writer.add_scalar('total_train_loss', train_loss, total_steps)

                    self._zero_optimizers()
                    train_loss.backward()

                    if opt_train.clip_grad:
                        max_norm = 1.0 if isinstance(opt_train.clip_grad, bool) else opt_train.clip_grad
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

                    self._step_optimizers(epoch)

                    epoch_train_loss += train_loss.item() * batch_size
                    num_items += batch_size
                    total_steps += 1
                    pbar.update(1)

                lrs = self._current_lrs()
                writer.add_scalar('base1_lr', lrs['base1'], total_steps)
                writer.add_scalar('base2_lr', lrs['base2'], total_steps)
                writer.add_scalar('detail_lr', lrs['detail'], total_steps)

                epoch_train_loss /= max(1, num_items)

                # Per-stage best
                if epoch_train_loss <= self._best_stage_loss:
                    self._best_stage_loss = epoch_train_loss
                    self._best_stage_epoch = epoch
                    save_checkpoint(
                        self, epoch, epoch_train_loss, lrs,
                        os.path.join(checkpoints_dir, f'best_{self._current_stage_name}.pth'),
                    )

                # Overall best
                if epoch_train_loss <= best_train_loss:
                    best_train_loss = epoch_train_loss
                    best_train_epoch = epoch
                    save_checkpoint(
                        self, epoch, epoch_train_loss, lrs,
                        os.path.join(checkpoints_dir, 'best_model_train.pth'),
                    )

                boundary_name = self._boundary_checkpoint_name(epoch)
                if boundary_name is not None:
                    save_checkpoint(
                        self, epoch, epoch_train_loss, lrs,
                        os.path.join(checkpoints_dir, boundary_name),
                    )

                if not total_steps % opt_train.steps_til_summary:
                    msg = (
                        f"Epoch {epoch} | Phase {phase} | Mode {mode} | "
                        f"Iter:{total_steps}, Loss {epoch_train_loss:.4f}, "
                        f"B1_Lr {lrs['base1']:.8f}, B2_Lr {lrs['base2']:.8f}, D_Lr {lrs['detail']:.8f}\n"
                    )
                    for name, loss in losses.items():
                        msg += f"{name}(X{opt_train.loss[name].factor}): {loss.item():.4f}\n"
                    tqdm.write(msg)

                self._step_schedulers(epoch, epoch_train_loss)

            save_checkpoint(
                self, epoch, epoch_train_loss, self._current_lrs(),
                os.path.join(checkpoints_dir, 'model_final.pth'),
            )


def optimize_code(opt, model):
    raise NotImplementedError('optimize_code was not updated for staged training.')

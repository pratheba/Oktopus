"""Training utilities for the 3-branch NGCNetGrid model.

There are three branches (base1, base2, detail), each with its own
optimizer and LR scheduler. All three share a small set of modules
(curve embedding, type embedding, curve encoder, positional encoders),
but those shared parameters appear in the param-generator of whichever
branch is active, so they get updated regardless of which optimizer steps.
"""

import os
import numpy as np
import torch


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(self, epoch, loss, lrs, filename):
    """Save model + all three optimizers + all three schedulers.

    Args:
        self: Trainer instance (provides `.model`, `.optim_*`, `.scheduler_*`).
        epoch, loss: training metadata to record.
        lrs: dict with keys 'base1', 'base2', 'detail' (current LRs).
        filename: absolute path to write.
    """
    model_dict = {
        'model': self.model.state_dict(),

        'optimizer_base1':  self.optim_base1.state_dict(),
        'optimizer_base2':  self.optim_base2.state_dict(),
        'optimizer_detail': self.optim_detail.state_dict(),

        'scheduler_base1':  self.scheduler_base1.state_dict() if self.scheduler_base1 is not None else None,
        'scheduler_base2':  self.scheduler_base2.state_dict() if self.scheduler_base2 is not None else None,
        'scheduler_detail': self.scheduler_detail.state_dict() if self.scheduler_detail is not None else None,

        'lr_base1':  lrs.get('base1'),
        'lr_base2':  lrs.get('base2'),
        'lr_detail': lrs.get('detail'),

        'epoch': epoch,
        'loss': loss,
    }
    torch.save(model_dict, filename)


def _resolve_ckpt_path(checkpoint_dir, checkpoint):
    """Map a checkpoint label to a file path under `{checkpoint_dir}/train/checkpoints/`.

    Labels: 'final', 'post', 'train' -> standard file names.
    Anything else is treated as an explicit filename.
    """
    if checkpoint in ('final', 'post'):
        ckpt_name = f'model_{checkpoint}.pth'
    elif checkpoint == 'train':
        ckpt_name = 'best_model_train.pth'
    else:
        ckpt_name = checkpoint
    return os.path.join(checkpoint_dir, 'train', 'checkpoints', ckpt_name)


def load_checkpoint(
    model,
    checkpoint_dir,
    device,
    checkpoint='final',
    optim_base1=None,
    optim_base2=None,
    optim_detail=None,
    scheduler_base1=None,
    scheduler_base2=None,
    scheduler_detail=None,
    strict=False,
):
    """Load model + optimizers + schedulers from a checkpoint.

    Any optimizer/scheduler argument that is None is skipped. Returns a dict
    of training metadata.
    """
    #ckpt_path = _resolve_ckpt_path(checkpoint_dir, checkpoint)
    #ckpt = torch.load(ckpt_path, map_location=device)
    if checkpoint in ['final', 'post']:
        ckpt_name = f'model_{checkpoint}.pth'
    elif checkpoint == 'train':
        ckpt_name = 'best_model_train.pth'
    else:
        ckpt_name = checkpoint

    ckpt_path = os.path.join(checkpoint_dir, 'train', f'checkpoints/{ckpt_name}')
    ckpt = torch.load(ckpt_path, map_location=device)

    # model
    model.load_state_dict(ckpt['model'], strict=strict)

    # optimizers
    if optim_base1 is not None and 'optimizer_base1' in ckpt:
        try:
            optim_base1.load_state_dict(ckpt['optimizer_base1'])
        except Exception as e:
            print(f"[resume] skipped optimizer_base1: {e}", flush=True)

    if optim_base2 is not None and 'optimizer_base2' in ckpt:
        try:
            optim_base2.load_state_dict(ckpt['optimizer_base2'])
        except Exception as e:
            print(f"[resume] skipped optimizer_base2: {e}", flush=True)

    if optim_detail is not None and 'optimizer_detail' in ckpt:
        try:
            optim_detail.load_state_dict(ckpt['optimizer_detail'])
        except Exception as e:
            print(f"[resume] skipped optimizer_detail: {e}", flush=True)

    # schedulers
    if scheduler_base1 is not None and 'scheduler_base1' in ckpt:
        try:
            scheduler_base1.load_state_dict(ckpt['scheduler_base1'])
        except Exception as e:
            print(f"[resume] skipped scheduler_base1: {e}", flush=True)

    if scheduler_base2 is not None and 'scheduler_base2' in ckpt:
        try:
            scheduler_base2.load_state_dict(ckpt['scheduler_base2'])
        except Exception as e:
            print(f"[resume] skipped scheduler_base2: {e}", flush=True)

    if scheduler_detail is not None and 'scheduler_detail' in ckpt:
        try:
            scheduler_detail.load_state_dict(ckpt['scheduler_detail'])
        except Exception as e:
            print(f"[resume] skipped scheduler_detail: {e}", flush=True)





def load_model(model, device, checkpoint_path, checkpoint='final'):
    """Load only the model weights (no optimizer/scheduler state)."""
    path = _resolve_ckpt_path(checkpoint_path, checkpoint)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    return model


# ---------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------

def _split_decay_params(named_params, decay_prefixes, no_decay_substrings):
    """Split parameters into (decay, no_decay) lists.

    A param goes to `no_decay` if:
      - it ends with `.bias`, OR
      - any substring in `no_decay_substrings` appears in its name, OR
      - its name does not start with any of `decay_prefixes`.

    Otherwise it goes to `decay`.
    """
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if name.endswith('.bias'):
            no_decay.append(p)
            continue
        if any(s in name for s in no_decay_substrings):
            no_decay.append(p)
            continue
        if any(name.startswith(pfx) for pfx in decay_prefixes):
            decay.append(p)
        else:
            no_decay.append(p)
    return decay, no_decay


def _build_optimizer(op_cfg, params_iter, named_params_iter,
                     decay_prefixes, no_decay_substrings,
                     weight_decay_default=1e-4):
    """Build one optimizer (Adam / AdamW / SGD) from an op_cfg block."""
    otype = op_cfg.type

    if otype == 'Adam':
        return torch.optim.Adam(
            params=list(params_iter),
            lr=op_cfg.lr,
            betas=(op_cfg.beta1, op_cfg.beta2),
            amsgrad=op_cfg.amsgrad,
        )

    if otype == 'AdamW':
        decay, no_decay = _split_decay_params(
            named_params_iter, decay_prefixes, no_decay_substrings,
        )
        wd = getattr(op_cfg, 'weight_decay', weight_decay_default)
        return torch.optim.AdamW(
            [
                {'params': decay,    'weight_decay': wd},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=op_cfg.lr,
            betas=(op_cfg.beta1, op_cfg.beta2),
            amsgrad=op_cfg.amsgrad,
        )

    if otype == 'SGD':
        return torch.optim.SGD(
            list(params_iter),
            lr=op_cfg.lr,
            momentum=getattr(op_cfg, 'momentum', 0.9),
        )

    raise NotImplementedError(f'Not implemented optimizer type: {otype}')


def _build_scheduler(op_cfg, optim):
    """Build a single LR scheduler from an op_cfg block. Returns None if
    no scheduler is configured."""
    sch_type = getattr(op_cfg, 'lr_scheduler', None)
    if not sch_type:
        return None

    if sch_type == 'MultiStep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=op_cfg.milestones, gamma=op_cfg.gamma,
        )
    if sch_type == 'ROP':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=op_cfg.factor, patience=op_cfg.patience,
        )
    if sch_type == 'CLR':
        return torch.optim.lr_scheduler.CyclicLR(
            optim, base_lr=op_cfg.base_lr, max_lr=op_cfg.max_lr,
            step_size_up=op_cfg.step,
        )
    raise NotImplementedError(f'Not implemented lr scheduler: {sch_type}')


def get_optimizer(opt, model):
    """Build three optimizers + schedulers for the 3-branch model.

    Expects `opt` to be a DottedDict-like with keys `optim_base1`,
    `optim_base2`, `optim_detail`. Expects `model` to expose the
    generators `base1_parameters`, `base2_parameters`, `detail_parameters`,
    plus their `named_*` counterparts.
    """
    op_b1  = opt.optim_base1
    op_b2  = opt.optim_base2
    op_det = opt.optim_detail

    # For AdamW weight-decay splits we decay FiLM + decoder weights only;
    # everything else (grids, embeddings, positional encoders, biases)
    # goes to the no-decay bucket.
    no_decay_substrings = ('grid_curve', '.grid', '.grids', 'embd',
                           'pos_enc', 'period_angle_enc')

    optim_base1 = _build_optimizer(
        op_b1,
        params_iter=model.base1_parameters(),
        named_params_iter=model.named_base1_parameters(),
        decay_prefixes=('base_model.filmenc_base1', 'base_model.decoder_base1'),
        no_decay_substrings=no_decay_substrings,
    )
    optim_base2 = _build_optimizer(
        op_b2,
        params_iter=model.base2_parameters(),
        named_params_iter=model.named_base2_parameters(),
        decay_prefixes=('base_model.filmenc_base2', 'base_model.decoder_base2'),
        no_decay_substrings=no_decay_substrings,
    )
    optim_detail = _build_optimizer(
        op_det,
        params_iter=model.detail_parameters(),
        named_params_iter=model.named_detail_parameters(),
        decay_prefixes=('detail_model.filmenc_detail', 'detail_model.decoder_detail'),
        no_decay_substrings=no_decay_substrings,
    )

    res = {
        'optimizer_base1':  optim_base1,
        'optimizer_base2':  optim_base2,
        'optimizer_detail': optim_detail,
        'epoch_lr_base1':  _build_scheduler(op_b1,  optim_base1),
        'epoch_lr_base2':  _build_scheduler(op_b2,  optim_base2),
        'epoch_lr_detail': _build_scheduler(op_det, optim_detail),
        # legacy keys (kept None but present so old callers don't KeyError)
        'epoch_lr': None,
        'step_lr':  None,
    }
    print('optimizer setup:', {k: type(v).__name__ for k, v in res.items()}, flush=True)
    return res


def get_lr_scheduler(optim, opt):
    """Standalone LR scheduler builder. Returns a scheduler or raises."""
    sch_type = opt['type']
    if sch_type == 'MultiStep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=opt['milestones'], gamma=opt['gamma'],
        )
    if sch_type == 'ROP':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=opt['factor'], patience=opt['patience'],
        )
    raise NotImplementedError(f'Not implemented lr scheduler: {sch_type}')

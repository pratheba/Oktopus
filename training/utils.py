import os
import numpy as np
import torch

def save_checkpoint(self, epoch, loss, base_lr, detail_lr, filename):
        #print("self model = ", self.model)
        model_dict = {}
        model_dict['model'] = self.model.state_dict()
        model_dict['optimizer_base'] = self.optim_base.state_dict()
        model_dict['optimizer_detail'] = self.optim_detail.state_dict()
        model_dict['scheduler_base'] = self.scheduler_base.state_dict()
        model_dict['scheduler_detail'] = self.scheduler_detail.state_dict()
        model_dict['lr_base'] = base_lr 
        model_dict['lr_detail'] = detail_lr 
        model_dict['epoch'] = epoch
        model_dict['loss'] = loss
        #if withval:
        #    model_dict['validator'] = self.validator.state_dict()

        torch.save(model_dict, filename)


def load_checkpoint(
    model,
    checkpoint_dir,
    device,
    checkpoint='final',
    optim_base=None,
    optim_detail=None,
    scheduler_base=None,
    scheduler_detail=None,
    strict=False,
):
    if checkpoint in ['final', 'post']:
        ckpt_name = f'model_{checkpoint}.pth'
    elif checkpoint == 'train':
        ckpt_name = 'best_model_train.pth'
    else:
        # either explicit filename or epoch-style name
        ckpt_name = checkpoint

    ckpt_path = os.path.join(checkpoint_dir, 'train', f'checkpoints/{ckpt_name}')
    ckpt = torch.load(ckpt_path, map_location=device)

    # model
    model.load_state_dict(ckpt['model'], strict=strict)

    # optimizers
    if optim_base is not None and 'optimizer_base' in ckpt:
        optim_base.load_state_dict(ckpt['optimizer_base'])

    if optim_detail is not None and 'optimizer_detail' in ckpt:
        optim_detail.load_state_dict(ckpt['optimizer_detail'])

    # schedulers
    if scheduler_base is not None and 'scheduler_base' in ckpt:
        scheduler_base.load_state_dict(ckpt['scheduler_base'])

    if scheduler_detail is not None and 'scheduler_detail' in ckpt:
        scheduler_detail.load_state_dict(ckpt['scheduler_detail'])

    # metadata
    epoch = ckpt.get('epoch', None)
    loss = ckpt.get('loss', None)
    lr_base = ckpt.get('lr_base', None)
    lr_detail = ckpt.get('lr_detail', None)

    return {
        'epoch': epoch,
        'loss': loss,
        'lr_base': lr_base,
        'lr_detail': lr_detail,
        'checkpoint_path': ckpt_path,
    }

def load_model(model, device, checkpoint_path, checkpoint='final'): 
    cpu_device = torch.device('cpu')

    if checkpoint == 'final' or checkpoint == 'post':
        ckpt_name = f'model_{checkpoint}.pth'
    elif checkpoint == 'train':
        ckpt_name = f'best_model_{checkpoint}.pth'
    else:
        #ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)
        ckpt_name = checkpoint 

    checkpoint_path = os.path.join(checkpoint_path, 'train', f'checkpoints/{ckpt_name}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def get_optimizer(opt, model):
    op_base = opt.optim_base
    op_detail = opt.optim_detail
    res = {}
    decay, no_decay = [], []
    for name, p in model.named_base_parameters():
        if not p.requires_grad:
            continue

        # never decay biases
        if name.endswith(".bias"):
            no_decay.append(p)
            continue

        # never decay grid tensors (use TV/seam instead)
        if "base_grid_curve" in name or ".grid" in name or ".grids" in name:
            no_decay.append(p)
            continue

        # decay only FiLM + decoder (usually the best)
        if name.startswith("filmenc_base") or name.startswith("decoder_base"):
            decay.append(p)
        else:
            no_decay.append(p)  # embeddings/encoder typically don't need decay


    if op_base.type == 'Adam':
        optim_base = torch.optim.Adam(params=model.base_parameters(), 
                lr=op_base.lr, betas=(op_base.beta1, op_base.beta2), amsgrad=op_base.amsgrad)
    elif op_base.type == 'AdamW':
        optim_base = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],lr=op_base.lr, betas=(op_base.beta1, op_base.beta2), amsgrad=op_base.amsgrad)
    elif op_base.type == 'SGD':
        optim_base = torch.optim.SGD(model.base_parameters(), lr=op_base.lr, momentum=op_base.momentum)

    if op_detail.type == 'Adam':
        optim_detail = torch.optim.Adam(params=model.detail_parameters(), 
                lr=op_detail.lr, betas=(op_detail.beta1, op_detail.beta2), amsgrad=op_detail.amsgrad)
    elif op_detail.type == 'AdamW':
        optim_detail = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, 
         {"params": no_decay, "weight_decay": 0.0}],lr=op_detail.lr, betas=(op_detail.beta1, op_detail.beta2), amsgrad=op_detail.amsgrad)
    elif op_detail.type == 'SGD':
        optim_detail = torch.optim.SGD(model.parameters(), lr=op_detail.lr, momentum=op_detail.momentum)
    else:
        raise NotImplementedError('Not implemented optimizer type')
    res['optimizer_base'] = optim_base
    res['optimizer_detail'] = optim_detail
    res['epoch_lr'] = None
    res['epoch_lr_base'] = None
    res['epoch_lr_detail'] = None
    res['step_lr'] = None
    if op_base.lr_scheduler:
        if op_base.lr_scheduler == 'MultiStep':
            lr_sch = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=op_base.milestones, gamma=op_base.gamma)
            res['epoch_lr'] = lr_sch
        elif op_base.lr_scheduler == 'ROP':
            lr_sch_base = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_base, factor=op_base.factor, patience=op_base.patience)
            res['epoch_lr_base'] = lr_sch_base
            lr_sch_detail = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_detail, factor=op_detail.factor, patience=op_detail.patience)
            res['epoch_lr_detail'] = lr_sch_detail
        elif op_base.lr_scheduler == 'CLR':
            lr_sch = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=p.base_lr, max_lr=p.max_lr, step_size_up=p.step)
            res['step_lr'] = lr_sch
        else:
            raise NotImplementedError('Not implemented lr scheduler')
    print("res = ", res, flush=True)
    return res


def get_lr_scheduler(optim, opt):
    sch_type = opt['type']
    if sch_type == 'MultiStep':
        lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=opt['milestones'], gamma=opt['gamma'])
    else:
        raise NotImplementedError



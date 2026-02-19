import os
import numpy as np
import torch

def save_checkpoint(self, epoch, loss, filename):
        #print("self model = ", self.model)
        model_dict = {}
        model_dict['model'] = self.model.state_dict()
        model_dict['optimizer_base'] = self.optim_base.state_dict()
        model_dict['optimizer_detail'] = self.optim_detail.state_dict()
        model_dict['scheduler_base'] = self.scheduler_base.state_dict()
        model_dict['scheduler_detail'] = self.scheduler_detail.state_dict()
        model_dict['epoch'] = epoch
        model_dict['loss'] = loss
        #if withval:
        #    model_dict['validator'] = self.validator.state_dict()

        torch.save(model_dict, filename)

def load_model(model, device, checkpoint_path, checkpoint='final'): 
    cpu_device = torch.device('cpu')

    if checkpoint == 'final' or checkpoint == 'post':
        ckpt_name = f'model_{checkpoint}.pth'
    else:
        #ckpt_name = 'model_epoch_%04d.pth' % int(checkpoint)
        ckpt_name = checkpoint 

    checkpoint_path = os.path.join(checkpoint_path, f'checkpoints/{ckpt_name}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def get_optimizer(opt, model):
    p = opt.optim_base
    res = {}
    if p.type == 'Adam':
        optim_base = torch.optim.Adam(params=model.base_parameters(), 
                lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad)
        optim_detail = torch.optim.Adam(params=model.detail_parameters(), 
                lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad)
    elif p.type == 'AdamW':
        #optim =  torch.optim.AdamW(params=model.parameters(), 
        #        lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad)
        optim_base = torch.optim.AdamW(params=model.base_parameters(), 
                lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad)
        optim_detail = torch.optim.AdamW(params=model.detail_parameters(), 
                lr=p.lr, betas=(p.beta1, p.beta2), amsgrad=p.amsgrad)
    elif p.type == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=p.lr, momentum=p.momentum)
    else:
        raise NotImplementedError('Not implemented optimizer type')
    res['optimizer_base'] = optim_base
    res['optimizer_detail'] = optim_detail
    res['epoch_lr'] = None
    res['epoch_lr_base'] = None
    res['epoch_lr_detail'] = None
    res['step_lr'] = None
    if p.lr_scheduler:
        if p.lr_scheduler == 'MultiStep':
            lr_sch = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=p.milestones, gamma=p.gamma)
            res['epoch_lr'] = lr_sch
        elif p.lr_scheduler == 'ROP':
            lr_sch_base = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_base, factor=p.factor, patience=p.patience)
            res['epoch_lr_base'] = lr_sch_base
            lr_sch_detail = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_detail, factor=p.factor, patience=p.patience)
            res['epoch_lr_detail'] = lr_sch_detail
        elif p.lr_scheduler == 'CLR':
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



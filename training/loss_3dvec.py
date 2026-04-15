import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class LossHandler():
    def __init__(self):
        self.metric_types = ['BCE', 'L1', 'L2', 'cos_sim', 'None']

    def get_metric_fn(self, opt):
        metric = opt['metric']
        if metric not in self.metric_types:
            raise NameError('Not supported metric')

        loss = None
        if metric == 'BCE':
            loss = nn.BCELoss()
        elif metric == 'L1':
            loss = nn.L1Loss()
        elif metric == 'L2':
            loss = nn.MSELoss()
        elif metric == 'cos_sim':
            loss = nn.CosineSimilarity(eps=1e-6)
        elif metric == 'None':
            loss = None
        else:
            raise NotImplementedError('Not implemented metric')

        return loss

    def parse_config(self, loss_schedule):
        self.loss_fn = {}
        for name, loss_config in loss_schedule.items():
            if loss_config['enable']:
                metric_fn = self.get_metric_fn(loss_config)
                self.loss_fn[name] = {
                    'metric_fn': metric_fn, 
                    'factor': loss_config['factor']
                }
                if 'clamp' in loss_config:
                    self.loss_fn[name]['clamp'] = loss_config['clamp']

    def soft_trunc(self,x, tau): 
        return tau * torch.tanh(x / tau)

    def sdf_base_loss(self, output, gt, metric_fn, clamp=0.1, epoch=0, E0=2000, E1=4500):
        out_sdf = output['sdf']
        #out_detail_sdf = output['sdf_detail']
        B, n = out_sdf.shape
        gt_sdf = gt['sdf'][:,:n].view_as(out_sdf)
        return metric_fn(out_sdf, gt_sdf)


    def sdf_loss(self, output, gt, metric_fn, clamp=0.1, epoch=0, E0=2000, E1=4500):
        out_sdf = output['sdf']
        out_base_sdf = output['sdf_base']
        out_detail_sdf = output['sdf_res']
        out_detail_res = output['detail_res']

        gt_sdf = gt['sdf'].view_as(out_sdf)
        gt_base_sdf = gt['sdf_base'].view_as(out_base_sdf)

        if epoch <= E0:
            return metric_fn(out_base_sdf, gt_base_sdf)
        elif epoch < E1:
            gt_residual_sdf = gt['sdf_res'].view_as(out_detail_sdf)
            mask = torch.abs(out_base_sdf.detach()) < 0.1
            return metric_fn(out_sdf, gt_sdf) + 0.05 * metric_fn(out_detail_res[mask], gt_residual_sdf[mask])
        else:
            gt_residual_sdf = gt['sdf_res'].view_as(out_detail_sdf)
            mask = torch.abs(out_base_sdf.detach()) < 0.1
            return metric_fn(out_sdf, gt_sdf) + 0.05*metric_fn(out_base_sdf, gt_base_sdf) + 0.05*metric_fn(out_detail_res[mask], gt_residual_sdf[mask])
 
    def sdf_query_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        gt_sdf = gt['sdf'].view_as(out_sdf)
        return metric_fn(out_sdf, gt_sdf)

    def sdf_context_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        gt_sdf = gt['sdf'].view_as(out_sdf)
        return metric_fn(out_sdf, gt_sdf)

    def code_loss(self, output, gt, metric_fn, epoch=0, E0=2000, E1=4500):
        #code = output['curve_code']
        code = output['code'] 
        reg_loss = torch.sum(torch.pow(code, 2), dim=-1)
        return torch.mean(reg_loss)

    def feature_loss(self, output, gt, metric_fn):
        enc_features = output['enc_features']
        dec_features = output['dec_features']
        return metric_fn(enc_features, dec_features)
        #reg_loss = torch.sum(torch.pow(code, 2), dim=-1)
        #return torch.mean(reg_loss)
    def eikonal_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        in_sample = gt['samples']
        B, n = out_sdf.shape
        B, n, d = in_sample.shape
        #gt_sdf = gt['sdf'][:,:n].view_as(out_sdf)
        grad = torch.autograd.grad(
            outputs = out_sdf,
            inputs = in_sample,
            grad_outputs = torch.ones_like(out_sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]

        grad_norm = torch.sqrt(torch.sum(grad * grad, dim=-1) + 1e-12)
        loss = (grad_norm - 1.0) ** 2
        return loss.mean() 
    
    def tv_loss_chw(self, G, isrho=False):
        tv_s    = (G[:, 1:, :] - G[:, :-1, :]).abs().mean()
        tv_t    = (G[:, :, 1:] - G[:, :, :-1]).abs().mean()
        if isrho:
            return (tv_s + tv_t)
        tv_wrap = (G[:, :, 0]  - G[:, :, -1]).abs().mean()
        return (tv_s + tv_t + tv_wrap)

    def tv_loss_bchw(self, G, isrho=False):
        tv_s    = (G[:, :, 1:, :] - G[:, :, :-1, :]).abs().mean()
        tv_t    = (G[:, :, :, 1:] - G[:, :, :, :-1]).abs().mean()
        if isrho:
            return (tv_s + tv_t)
        tv_wrap = (G[:, :, :, 0]  - G[:, :, :, -1]).abs().mean()
        return (tv_s + tv_t + tv_wrap)

    def tv_l2_grid_loss(self, grids, factor, isrho=False):
        loss = 0.0
        growth = 10.0
        for level, grid in enumerate(grids):
            w = factor * (growth ** level)
            loss = loss + w * (grid**2).mean()
        return loss

    def tv_grid_loss(self, grids, factor, isrho=False):
        loss = 0.0
        alpha = 0.5
        for g in grids:   # coarsest -> finest
            if g.dim() == 3:   # (C,H,W)
                loss = loss + factor * self.tv_loss_chw(g, isrho)
                #loss = loss + self.tv_loss_chw(g, isrho)
            elif g.dim() == 4: # (1,C,H,W) or (B,C,H,W)
                loss = loss + factor * self.tv_loss_bchw(g, isrho)
                #loss = loss + self.tv_loss_bchw(g, isrho)
            else:
                raise ValueError(f"Unexpected grid dim: {g.shape}")
            factor *= alpha
        return loss

    def tv_grid_ctloss(self, output, factor=1e-4):
        grids = output['base_grid_ct']
        return self.tv_grid_loss(grids, factor)
 
    def tv_grid_crloss(self, output, factor=1e-4):
        grids = output['base_grid_cr']
        return self.tv_grid_loss(grids, factor, True)

    def tv_l2_grid_ctloss(self, output, factor=1e-4):
        grids = output['base_grid_ct']
        return self.tv_l2_grid_loss(grids, factor)
 
    def tv_l2_grid_crloss(self, output, factor=1e-4):
        grids = output['base_grid_cr']
        return self.tv_l2_grid_loss(grids, factor, True)

    def lipschitz_loss(self, output, factor):
        out_sdf = output['sdf']
        B, n = out_sdf.shape
        gt_sdf = gt['sdf'][:,:n].view_as(out_sdf)
        out_sdf = out_sdf.reshape(B*n)
        
        return metric_fn(out_sdf, gt_sdf)
    
    
    def nodes_loss(self, output, gt, metric_fn):
        nodes = output['nodes']
        nodes_gt = gt['nodes']

        return metric_fn(nodes, nodes_gt)
    
    def mat_loss(self, output, gt, metric_fn):
        mat = output['mat']
        mat_gt = gt['mat']

        return metric_fn(mat, mat_gt)
    
    def kl_loss(self, output, gt, metric_fn):
        # already calculated in model
        kl_reg = output['kl']
        return torch.mean(kl_reg)
    
    def edm_loss(self, output, gt, metric_fn):
        loss = output['loss']
        return loss.mean()
    
    def occ_sum_loss(self, output, gt, metric_fn):
        # pred(Nc, Ns), gt(Ns)
        occ_pred = output['occ']
        occ_gt = gt['occ']

        occ_sum = torch.sum(occ_pred, dim=0)
        return metric_fn(occ_sum, occ_gt)
    
    def occ_max_loss(self, output, gt, metric_fn):
        # pred(Nc, Ns), gt(Ns)
        occ_pred = output['occ']
        occ_gt = gt['occ']

        occ_max,_ = torch.max(occ_pred, dim=0)
        return metric_fn(occ_max, occ_gt)
    

    def __call__(self, output, gt, epoch, E0, E1):
        res = {}
        for name, loss in self.loss_fn.items():
            if hasattr(self, name):
                func = getattr(self, name)
            else:
                raise NameError(f'Not defined {loss} {name}')
            if 'tv' in name:
                loss_term = func(output, loss['factor'])
                res[name] = loss_term
            elif 'sdf' in name:
                loss_term = func(output, gt, loss['metric_fn'], loss['clamp'], epoch, E0, E1)
                res[name] = loss['factor']*loss_term
            else:
                loss_term = func(output, gt, loss['metric_fn'], epoch, E0, E1)
                res[name] = loss['factor']*loss_term

        return res

def config_loss(loss_schedule):
    loss_handler = LossHandler()
    loss_handler.parse_config(loss_schedule)
    return loss_handler


def eval_loss(output, gt, loss_schedule):
    res = {}
    for name, config in loss_schedule.items():
        if config.enable:
            loss_term = getattr(loss, name)(output, gt, config)
            # print('name: {}, loss: {}'.format(name, loss_term))
            if isinstance(loss_term, dict):
                res.update(loss_term)
            else:
                res[name] = loss_term

    return res


def eval_val_loss(output, gt, output_type, metric='L1'):
    return getattr(loss, 'val_loss')(output, gt, output_type, metric)

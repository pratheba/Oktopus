
import torch
import torch.nn as nn


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

        if metric == 'BCE':
            return nn.BCELoss()
        elif metric == 'L1':
            return nn.L1Loss()
        elif metric == 'L2':
            return nn.MSELoss()
        elif metric == 'cos_sim':
            return nn.CosineSimilarity(eps=1e-6)
        elif metric == 'None':
            return None
        raise NotImplementedError('Not implemented metric')

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

    def _safe_masked_metric(self, pred, target, mask, metric_fn):
        if mask is None:
            return metric_fn(pred, target)
        if mask.sum().item() == 0:
            return pred.new_tensor(0.0)
        return metric_fn(pred[mask], target[mask])

    def sdf_base_loss(
        self, output, gt, metric_fn, clamp=0.1, epoch=0,
        E0=1000, E1=1500, E2=2000, E3=2500,
        lambda_b1=0.05, lambda_b2=0.05, mode='base_train'
    ):
        out_base  = output['sdf_base']
        out_base1 = output['sdf_base1']
        gt_base   = gt['sdf_base'].view_as(out_base)

        pred_base2_res = out_base - out_base1.detach()
        gt_base2_res   = gt_base - out_base1.detach()
        base_mask = torch.abs(out_base1.detach()) < clamp

        if mode == 'detail_train_from_base_ckpt':
            return metric_fn(out_base, gt_base)

        if mode == 'joint_finetune':
            pred_base2_res_joint = out_base - out_base1
            gt_base2_res_joint   = gt_base - out_base1
            return (
                metric_fn(out_base, gt_base)
                + lambda_b1 * metric_fn(out_base1, gt_base)
                + lambda_b2 * self._safe_masked_metric(
                    pred_base2_res_joint, gt_base2_res_joint, base_mask, metric_fn
                )
            )

        # base_train
        if epoch < E0:
            return metric_fn(out_base1, gt_base)
        elif epoch < E1:
            return self._safe_masked_metric(pred_base2_res, gt_base2_res, base_mask, metric_fn)
        else:
            pred_base2_res_joint = out_base - out_base1
            gt_base2_res_joint   = gt_base - out_base1
            return (
                metric_fn(out_base, gt_base)
                + lambda_b1 * metric_fn(out_base1, gt_base)
                + lambda_b2 * self._safe_masked_metric(
                    pred_base2_res_joint, gt_base2_res_joint, base_mask, metric_fn
                )
            )

    def sdf_loss(
        self, output, gt, metric_fn, clamp=0.1, epoch=0,
        E0=1000, E1=1500, E2=2000, E3=2500,
        lambda_b1=0.05, lambda_b2=0.05, lambda_d=0.05,
        mode='base_train'
    ):
        out_sdf       = output['sdf']
        out_base      = output['sdf_base']
        out_base1     = output['sdf_base1']
        out_base2     = output['sdf_base2']
        out_detail    = output['sdf_detail']
        gate_base2    = output['gate_base2']
        gate_detail   = output['gate_detail']

        gt_sdf  = gt['sdf'].view_as(out_sdf)
        gt_base = gt['sdf_base'].view_as(out_base)
        gt_res  = gt['sdf_res'].view_as(out_sdf)

        pred_base2_res = out_base - out_base1.detach()
        gt_base2_res   = gt_base - out_base1.detach()
        pred_detail_res = gate_detail * out_detail

        base_mask   = torch.abs(out_base1.detach()) < clamp
        detail_mask = torch.abs(out_base.detach()) < clamp

        if mode == 'detail_train_from_base_ckpt':
            #print(metric_fn(out_sdf, gt_sdf))
            #print(self._safe_masked_metric(
            #        pred_detail_res, gt_res, detail_mask, metric_fn), flush=True)
            full_term = metric_fn(out_sdf, gt_sdf)
            #res_term = self._safe_masked_metric(pred_detail_res, gt_res, detail_mask, metric_fn)
            #print("full_term:", full_term.item())
            #print("res_term :", res_term.item())
            #print("total    :", (0.5*full_term + 0.2 * res_term).item())
            #print("mask_frac:", detail_mask.float().mean().item())
            #print("gate_mean:", gate_detail.mean().item())
            #print("pred_detail_res_abs_mean:", pred_detail_res.abs().mean().item(), flush=True)
            #lambda_d = 0.2
            #return (
            #    full_term
            #    + lambda_d * res_term)
            return full_term

        if mode == 'joint_finetune':
    
            return (metric_fn(out_sdf, gt_sdf)
                + lambda_b1 * metric_fn(out_base, gt_base))
            #pred_base2_res_joint = out_base - out_base1
            #gt_base2_res_joint   = gt_base - out_base1
            #pred_detail_res_joint = out_sdf - out_base
#            return (
#                metric_fn(out_sdf, gt_sdf)
#                + lambda_b1 * metric_fn(out_base, gt_base)
#                + lambda_b2 * self._safe_masked_metric(
#                    pred_base2_res_joint, gt_base2_res_joint, base_mask, metric_fn
#                )
#                + lambda_d * self._safe_masked_metric(
#                    pred_detail_res_joint, gt_res, detail_mask, metric_fn
#                )
#            )

        # base_train
        if epoch < E0:
            return metric_fn(out_base1, gt_base)
        elif epoch < E1:
            return self._safe_masked_metric(pred_base2_res, gt_base2_res, base_mask, metric_fn)
        elif epoch < E2:
            pred_base2_res_joint = out_base - out_base1
            gt_base2_res_joint   = gt_base - out_base1
            return (
                metric_fn(out_base, gt_base)
                + lambda_b1 * metric_fn(out_base1, gt_base)
                + lambda_b2 * self._safe_masked_metric(
                    pred_base2_res_joint, gt_base2_res_joint, base_mask, metric_fn
                )
            )
        else:
            return (
                metric_fn(out_sdf, gt_sdf)
                + lambda_d * self._safe_masked_metric(
                    pred_detail_res, gt_res, detail_mask, metric_fn
                )
            )

    def sdf_query_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        gt_sdf = gt['sdf'].view_as(out_sdf)
        return metric_fn(out_sdf, gt_sdf)

    def code_loss(self, output, gt, metric_fn, epoch=0, E0=1000, E1=1500):
        code = output['code']
        reg_loss = torch.sum(torch.pow(code, 2), dim=-1)
        return torch.mean(reg_loss)

    def eikonal_loss(self, output, gt, metric_fn):
        out_sdf = output['sdf']
        in_sample = gt['samples']
        grad = torch.autograd.grad(
            outputs=out_sdf,
            inputs=in_sample,
            grad_outputs=torch.ones_like(out_sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = torch.sqrt(torch.sum(grad * grad, dim=-1) + 1e-12)
        loss = (grad_norm - 1.0) ** 2
        return loss.mean()

    def tv_loss_chw(self, G, isrho=False):
        tv_s = (G[:, 1:, :] - G[:, :-1, :]).abs().mean()
        tv_t = (G[:, :, 1:] - G[:, :, :-1]).abs().mean()
        if isrho:
            return (tv_s + tv_t)
        tv_wrap = (G[:, :, 0] - G[:, :, -1]).abs().mean()
        return (tv_s + tv_t + tv_wrap)

    def tv_loss_bchw(self, G, isrho=False):
        tv_s = (G[:, :, 1:, :] - G[:, :, :-1, :]).abs().mean()
        tv_t = (G[:, :, :, 1:] - G[:, :, :, :-1]).abs().mean()
        if isrho:
            return (tv_s + tv_t)
        tv_wrap = (G[:, :, :, 0] - G[:, :, :, -1]).abs().mean()
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
        for g in grids:
            if g.dim() == 3:
                loss = loss + factor * self.tv_loss_chw(g, isrho)
            elif g.dim() == 4:
                loss = loss + factor * self.tv_loss_bchw(g, isrho)
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

    def __call__(self, output, gt, epoch, E0, E1, E2, E3, mode='base_train'):
        res = {}
        for name, loss in self.loss_fn.items():
            if hasattr(self, name):
                func = getattr(self, name)
            else:
                raise NameError(f'Not defined {name}')

            if 'tv' in name:
                loss_term = func(output, loss['factor'])
                res[name] = loss_term
            elif 'sdf' in name:
                loss_term = func(
                    output, gt, loss['metric_fn'], loss.get('clamp', 0.1),
                    epoch, E0, E1, E2, E3, mode=mode
                )
                res[name] = loss['factor'] * loss_term
            else:
                loss_term = func(output, gt, loss['metric_fn'], epoch, E0, E1)
                res[name] = loss['factor'] * loss_term
        return res


def config_loss(loss_schedule):
    loss_handler = LossHandler()
    loss_handler.parse_config(loss_schedule)
    return loss_handler

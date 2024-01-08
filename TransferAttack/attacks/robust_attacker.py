import numpy as np
import random
import torch
from torch import nn
import scipy.stats as st
import torch.nn.functional as F
from . transfer_attacker import DI, T_kernel
from .awp import AdvWeightPerturb


def set_require_grad(model):
    for pp in model.model.parameters():
        pp.requires_grad = True


class RobustMinimaxAttacker():
    def __init__(self, radius, steps, step_size, reverse_radius, reverse_steps, reverse_step_size, ascending=True, awp=False, surrogate_grad=False, TI=False, DI=False, MI=False):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.reverse_radius = reverse_radius / 255.
        self.reverse_steps = reverse_steps
        self.reverse_step_size = reverse_step_size / 255.
        self.ascending = ascending

        self.norm_type = 'l-infty'
        self.late_start = 0

        self.loss_type = 'ce'
        self.DI = DI
        self.TI = TI
        self.MI = MI
        self.grad_pre = None
        self.adv_x = None
        self.awp = awp

        self.surrogate_grad = surrogate_grad
        if self.surrogate_grad:
            self.step_grad = None

    def get_perpendicular_grad(self, a, b):
        # project a into b, channel-wise
        height = a.shape[-2]
        width = a.shape[-1]
        channel = a.shape[-3]

        a = a.reshape(-1, a.shape[-1] * a.shape[-2] * a.shape[-3])
        b = b.reshape(-1, b.shape[-1] * b.shape[-2] * b.shape[-3])
        a1_proj_b1 = torch.stack(
            [b[i] * torch.vdot(a[i], b[i]) / torch.vdot(b[i], b[i]) for i in range(a.shape[0])])
        perpendicular_a = a - a1_proj_b1
        return perpendicular_a.reshape(-1, channel, height, width)

    def one_step(self, model, x, y, r_p):
        for pp in model.parameters():
            pp.requires_grad = False
        max_neihbor = self.adv_x.detach().clone() + r_p
        max_neihbor.requires_grad_()
        logits = model(DI(max_neihbor)) if self.DI else model(max_neihbor)  # DI
        if self.loss_type == 'logit' and not self.ascending:
            real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            logit_dists = (-1 * real)
            loss = logit_dists.mean()
        else:
            loss = nn.CrossEntropyLoss(reduction='mean')(logits, y)
        grad = torch.autograd.grad(loss, [max_neihbor])[0]
        max_neihbor.requires_grad_(False)
        if self.surrogate_grad:
            self.adv_x.requires_grad_()
            x_loss = nn.CrossEntropyLoss(reduction='mean')(model(self.adv_x), y)
            surrogate_grad = torch.autograd.grad(x_loss, [self.adv_x])[0]
            self.adv_x.requires_grad_(False)
            v_grad = self.get_perpendicular_grad(surrogate_grad, grad)
            grad -= 0.02 * v_grad

        with torch.no_grad():
            grad_c = grad.clone()
            if not self.ascending: grad_c.mul_(-1)
            if self.TI:
                grad_c = F.conv2d(grad_c, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)  # TI
            if self.MI:
                grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * self.grad_pre
                self.grad_pre = grad_c

            self.adv_x.add_(torch.sign(grad_c), alpha=self.step_size)
        self._clip_(self.adv_x, x)
        set_require_grad(model)

    def outer(self, model, adv_x, x, y, r_p):
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        grad_pre = 0
        for step in range(2):
            adv_x.requires_grad_()
            logits = model(DI(adv_x + r_p)) if self.DI else model(adv_x + r_p)  # DI
            if self.loss_type == 'logit' and self.ascending:
                real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
                logit_dists = (-1 * real)
                loss = logit_dists.sum()
            else:
                loss = nn.CrossEntropyLoss(reduction='sum')(logits, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]
            adv_x.requires_grad_(False)

            with torch.no_grad():
                grad_c = grad.clone()
                if not self.ascending: grad_c.mul_(-1)
                if self.TI:
                    grad_c = F.conv2d(grad_c, T_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
                if self.MI:
                    grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                    grad_pre = grad_c

                adv_x.add_(torch.sign(grad_c), alpha=self.step_size)
            self._clip_(adv_x, x)

        return adv_x

    def inner(self, model, adv_x, y):
        r_p = torch.zeros_like(adv_x)
        if self.reverse_steps == 0 or self.reverse_radius == 0:
            return r_p.clone()
        adv_x = adv_x.clone()
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False
        for step in range(self.reverse_steps):
            r_p.requires_grad_()
            logits = model(adv_x+r_p)

            if self.loss_type == 'logit' and self.ascending:
                real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
                logit_dists = (-1 * real)
                loss = logit_dists.sum()
            else:
                loss = nn.CrossEntropyLoss(reduction='mean')(logits, y)
            grad = torch.autograd.grad(loss, [r_p])[0]
            r_p.requires_grad_(False)

            with torch.no_grad():
                # the gradient direction is opposite
                if self.ascending: grad.mul_(-1)
                r_p.add_(torch.sign(grad), alpha=self.reverse_step_size)
                r_p.clamp_(-self.reverse_radius, self.reverse_radius)

                #  这里我remove掉 sign,  然后
                # r_p.add_(grad, alpha=0.09)
            # r_p.clamp_(-self.reverse_radius, self.reverse_radius)
        set_require_grad(model)
        return r_p.data

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return x.clone()

        self.grad_pre = torch.zeros_like(x)

        self.adv_x = torch.zeros_like(x)

        if self.awp:
            AWP = AdvWeightPerturb(model=model, eta=1e-3, nb_iter=1)
            # AWP = AdvWeightPerturb(model=model, eta=5e-3, nb_iter=5)

        for step in range(self.steps):
            if step >= self.late_start:
                if self.awp:
                    AWP.perturb(self.adv_x, y)
                r_p = self.inner(model, self.adv_x, y)
                if self.awp:
                    AWP.restore()
                self.one_step(model, x, y, r_p)

            else:
                r_p = torch.zeros_like(x)
                self.one_step(model, x, y, r_p)

        set_require_grad(model)
        return self.adv_x.data

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0],-1)**2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0],-1).abs().sum(dim=1)
            norm = norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        adv_x.clamp_(0, 1)


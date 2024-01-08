import random

import numpy as np
import torch.nn as nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

class CutMixCrossEntropyLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)


class APM(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, APM_gamma=0.0125, times=20, **kwargs):
        defaults = dict(**kwargs)
        super(APM, self).__init__(model.parameters(), defaults) #
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.model = model
        self.times = times
        self.gamma = APM_gamma

    @torch.no_grad()
    def noise_step(self):
        # save the model  fist
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
        # add gaussian noise to non-bn params
        for name, p in self.model.named_parameters():
            # if 'bn' not in name:
            # temp = torch.empty_like(p, device=p.data.device)
            # temp.normal_(0, self.gamma)
            # p.data.add_(temp)
            if len(p.shape) > 1:
                sh = p.shape
                sh_mul = np.prod(sh[1:])
                temp = p.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(p.shape)
                temp = torch.normal(0, self.gamma * temp).to(p.data.device)
            else:
                temp = torch.empty_like(p, device=p.data.device)
                temp.normal_(0, self.gamma * (p.view(-1).norm().item()))
            p.data.add_(temp)

    @torch.no_grad()
    def de_noise_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "requires closure, but it was not provided"
        # print(closure)
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        for time in range(self.times):
            self.noise_step()
            loss, logits = closure(times=self.times)
            self.de_noise_step()

        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None: continue
        #         p.grad.div_(self.times)
        self.base_optimizer.step()
        return loss, logits


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=True)
        loss, logits = closure()
        self.second_step()
        return loss, logits

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups






class LSAM(torch.optim.Optimizer):
    '''
    different from SAM, don't use backward before LSAM step, we only use one backward inside the LSAM step
    '''
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LSAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.last_d_p_norm = None


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self.last_d_p_norm
        if not grad_norm:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["old_p"] = p.data.clone()

        else:
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for p in group["params"]:
                    try:
                        last_d_p = self.state[p]["last_grad"]
                    except KeyError:
                        continue
                    self.state[p]["old_p"] = p.data.clone() # save current p
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * last_d_p * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        self.last_d_p_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["last_grad"] = p.grad.data.clone() # save current grad
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update use neighborhood

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=True)
        loss, logits = closure()
        self.second_step()
        return loss, logits

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm



class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, beta=1.0, gamma=1.0, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # first order sum
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True
                if p.grad is None: continue
                # original sam
                # e_w = p.grad * scale.to(p)
                # asam
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    '''
    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"
    '''

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self):
        inputs, targets, loss_fct, model, defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True

        logits = model(inputs)
        loss = loss_fct(logits, targets)

        l_before = loss.clone().detach()
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()
        defined_backward(loss)

        # first step to w + e(w)
        self.first_step(True)

        with torch.no_grad():
            l_after = loss_fct(model(inputs), targets)
            instance_sharpness = l_after - l_before

            # codes for sorting
            prob = self.gamma
            if prob >= 0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff, _ = torch.topk(instance_sharpness, position)
                cutoff = cutoff[-1]

                # cutoff = 0
                # select top k%

                indices = [instance_sharpness > cutoff]

                # second forward-backward step
        # self.first_half()

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False

        loss = loss_fct(model(inputs[indices]), targets[indices])
        loss = loss.mean()
        defined_backward(loss)
        self.second_step(True)

        return predictions, return_loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                # original sam
                # p.grad.norm(p=2).to(shared_device)
                # asam
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm



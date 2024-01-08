import random

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.nn.utils import parameters_to_vector

##define TI
"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

T_kernel = gkern(5,3)

##define DI
def DI(x,  diversity_prob=0.7):
    img_size = x.shape[-1]
    resize_rate = random.uniform(0.75, 1.5)
    img_resize = int(img_size * resize_rate)

    if img_resize == img_size:
        return x
    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    # padded = F.interpolate(padded, size=[x.shape[-1], x.shape[-1]], mode='bilinear', align_corners=False)
    return padded if torch.rand(1) < diversity_prob else x


class TransferAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type,
                 TI=False, MI=False, DI=False, ascending=True, loss_type='ce', use_LGV=False, model_path=None, normalize=None):
        self.radius = radius / 255.
        self.steps = steps
        self.step_size = step_size / 255.
        self.random_start = random_start
        self.norm_type = norm_type
        self.ascending = ascending
        self.loss_type = loss_type
        self.TI = TI
        self.MI = MI
        self.DI = DI
        self.LGV = use_LGV
        if self.LGV:
            self.model_list = self.get_model_list(model_path) if model_path else None
            self.normalize = normalize

        if self.loss_type == 'trajectory':
            self.trajectory_logits_queue = deque([])
            self.max_queue_length = 6
            self.lamda = 0.01
            self.tao = 5

        if self.loss_type == 'doubleBP':
            self.beta = 0.05

    def get_model_list(self, model_path):
        import os
        all_list = list()
        file_list = os.listdir(model_path)
        for file in file_list:
            if '-model.pkl' in file:
                all_list.append(os.path.join(model_path, file))
        return all_list

    def perturb(self, model, criterion, x, y):
        if self.steps==0 or self.radius==0:
            return x.clone()
        delta = torch.zeros_like(x, requires_grad=False)

        if self.random_start:
            if self.norm_type == 'l-infty':
                delta.data = 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                delta.data = 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        delta.requires_grad_()

        grad_pre = 0
        if self.loss_type == 'doubleBP':
            for pp in model.parameters():
                pp.requires_grad = True
## TODO: 是不是要改成2范数试试看，参考Distill
        for step in range(self.steps):
            if self.LGV:
                model_path = random.choice(self.model_list)
                state_dict = torch.load(model_path, map_location=torch.device('cuda:0'))
                model.model.load_state_dict(state_dict['model_state_dict'])
                del state_dict
            logits = model(DI(x+delta)) if self.DI else model(x+delta)  # DI
            if self.loss_type == 'logit':
                real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
                logit_dists = (-1 * real)
                loss = logit_dists.sum()
            elif self.loss_type == 'trajectory':
                loss = nn.CrossEntropyLoss(reduction='sum')(logits, y)

                self.trajectory_logits_queue.append(logits.data)
                if len(self.trajectory_logits_queue) >= self.max_queue_length:
                    history_logits = self.trajectory_logits_queue.popleft()
                    trajectory_loss = nn.KLDivLoss(reduction='sum')(history_logits / self.tao, logits / self.tao )
                    loss -= self.lamda * trajectory_loss
            elif self.loss_type == 'doubleBP':
                self.ascending = False  # force to do gradient descent, not ascend
                loss = nn.CrossEntropyLoss(reduction='sum')(logits, y)
                grads = torch.autograd.grad(loss, inputs=logits, create_graph=True)
                grads = parameters_to_vector(grads)
                loss -= self.beta * torch.norm(grads)
            else:
                loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()

            with torch.no_grad():
                grad_c = delta.grad.clone()
                if not self.ascending: grad_c.mul_(-1)
                if self.TI:
                    grad_c = F.conv2d(grad_c, T_kernel, bias=None, stride=1,  padding='same', groups=3)  # TI
                if self.MI:
                    grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 0.9 * grad_pre  # MI
                grad_pre = grad_c
                delta.grad.zero_()
                delta.data = delta.data.add(torch.sign(grad_c), alpha=self.step_size)
                delta.data = delta.data.clamp(-self.radius, self.radius)
                delta.data = ((x + delta.data).clamp(0, 1)) - x

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True
        return x + delta.data


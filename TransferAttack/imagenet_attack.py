"""Implementation of sample attack."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random

from imagenet_csv import ImageNet
from CIFAR_Train.utils import get_arch

import pandas as pd
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import attacks
from attacks.transfer_attacker import DI, T_kernel
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from attacks import Normalize
from torch.utils.data import DataLoader, Dataset
import argparse
import pretrainedmodels
import timm
import torchattacks


IMAGE_SIZE = 224

# =[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225]


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./dataset/imagenet/outputs-resnet50/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=IMAGE_SIZE, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=IMAGE_SIZE, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=50, help="How many images process at one time.")
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
parser.add_argument('--targeted', action='store_true', default=False)
parser.add_argument('--DI', action='store_true', default=False)
parser.add_argument('--TI', action='store_true', default=False)
parser.add_argument('--MI', action='store_true', default=False)

opt = parser.parse_args()

TARGETED = opt.targeted

transforms = T.Compose(
    [T.Resize(IMAGE_SIZE), T.ToTensor()]
)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


def Spectrum_Simulation_Attack(images, gt, model, min, max):
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation
    :param max: the max the clip operation
    :return: the adversarial images
    """
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = opt.num_iter_set
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = 0
    rho = opt.rho
    N = opt.N
    sigma = opt.sigma

    for i in range(num_iter):
        noise = 0
        for n in range(N):
            gauss = torch.randn(x.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = attacks.dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = attacks.idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978


            output_v3 = model(x_idct) if not opt.DI else  model(DI(x_idct))

            loss = F.cross_entropy(output_v3, gt)
            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        if opt.MI :
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


class WrapModel(torch.nn.Module):
    def __init__(self, normalize, model):
        super(WrapModel, self).__init__()
        self.model = model
        self.normal = normalize

    def forward(self, input):
        return self.model(self.normal(input))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def main():
    seed_everything(42)
    model = timm.models.create_model('resnet50', pretrained=False)
    if opt.resume_path is not None:
        state_dict = torch.load(opt.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        del state_dict

    model = WrapModel(Normalize(opt.mean, opt.std), model).cuda()

    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    defender = attacks.TransferAttacker(
        radius=opt.max_epsilon,
        steps=opt.num_iter_set,
        step_size=2,
        norm_type= 'l-infty',
        ascending= not TARGETED,
        random_start=False,
        loss_type='ce',
        TI=opt.TI,
        DI=opt.DI,
        MI=opt.MI,
    )
    defender = attacks.PGDAttacker(
        radius=opt.max_epsilon,
        steps=opt.num_iter_set,
        step_size=2,
        norm_type='l-infty',
        ascending=not TARGETED,
        random_start=True,
    )



    for images, images_ID,  gt_real, gt_target in tqdm(data_loader):

        if TARGETED:
            gt = (gt_target).cuda()
        else:
            gt = (gt_real) .cuda()
        images = images.cuda()

        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        if TARGETED:
            adv_img = defender.perturb(model, torch.nn.functional.cross_entropy, images, gt)
        else:
            adv_img = Spectrum_Simulation_Attack(images, gt, model, images_min, images_max)
        print(torch.argmax(model(images), dim=1) == gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()

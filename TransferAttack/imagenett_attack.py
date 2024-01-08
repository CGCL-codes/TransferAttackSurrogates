"""Implementation of sample attack."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import timm
from CIFAR_Train import utils
import random

import torchvision.datasets

import pandas as pd
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import attacks
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from attacks import Normalize
from torch.utils.data import DataLoader
import argparse
import timm
import torchattacks

IMAGE_SIZE = 224

# =[0.485, 0.456, 0.40], std=[0.229, 0.224, 0.225]


parser = argparse.ArgumentParser()
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=IMAGE_SIZE, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=IMAGE_SIZE, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=50, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument('--test_dir', type=str, default='../Datasets/imagenette/val', help='load imagenett')
parser.add_argument('--resume_path', type=str, default=None, help='load models')
parser.add_argument('--output_dir', type=str, default='./imagenett-adv/resnet50-8_255/val', help='save imagenett adv')
parser.add_argument('--adv_type', type=str, default='pgd')
parser.add_argument('--targeted', action='store_true', default=False)


opt = parser.parse_args()
TARGETED = opt.targeted

transforms = T.Compose(
    [T.Resize([256,256]), T.CenterCrop([224,224]), T.ToTensor()]
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


def save_image(images, names, output_dir):
    """save the adversarial images"""

    for i, path in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        name = path.split('val/')[-1]
        if not os.path.exists(os.path.join(output_dir, name.split('/')[0])):
            os.makedirs(os.path.join(output_dir, name.split('/')[0]))
        img.save(os.path.join(output_dir , name))


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
    num_iter = 10
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
            x_idct = V(x_idct, requires_grad=True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            # output_v3 = model(DI(x_idct))

            output_v3 = model(x_idct)

            loss = F.cross_entropy(output_v3, gt)
            loss.backward()
            noise += x_idct.grad.data
        noise = noise / N

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        # noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        # noise = momentum * grad + noise
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


class ImageNette(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super(ImageNette, self).__init__(root=root, transform=transform)
        self.transform = transform
        # self.samples = random.sample(self.samples, 1000)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def verify(arch,arch_path,output_dir,batch_size, on_imagenet=False):
    if on_imagenet:
        model = utils.get_arch(arch, 'imagenet', pretrained=True)
        normalize = utils.get_normalize('imagenet')

    else:
        model = utils.get_arch(arch, 'imagenett')
        state_dict = torch.load(arch_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        del state_dict
        normalize = utils.get_normalize('imagenett')
    model = WrapModel(model=model,normalize=normalize).cuda()

    if on_imagenet:
        # imagenet_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        imagenet_index = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

        def trans(y):
            return imagenet_index[(y + 1) % 10] if TARGETED else imagenet_index[y]

        X = torchvision.datasets.ImageFolder(output_dir, transform=transforms, target_transform=trans)
    else:
        imagenet_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        def trans(y):
            return imagenet_index[(y + 1) % 10] if TARGETED else imagenet_index[y]
        X = torchvision.datasets.ImageFolder(output_dir, transform=transforms, target_transform=trans)

    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum_real = 0.
    loss_list = []

    for images, label in data_loader:
        label = label.cuda()
        images = images.cuda()
        with torch.no_grad():
            if TARGETED:
                sum_real += (torch.argmax(model(images), dim=1) == label).detach().sum().cpu()
            else:
                sum_real += (torch.argmax(model(images), dim=1) != label).detach().sum().cpu()
            # loss_list.append(torch.nn.functional.cross_entropy(model(images), label).item())
    if TARGETED:
        print(arch + '  targeted ASR = {:.2%}, targeted Loss = {:.2f}'.format(sum_real / len(X), sum(loss_list) / len(X)))
    else:
        print(arch + '  Untargeted ASR = {:.2%}, Untargeted Loss = {:.2f}'.format(sum_real / len(X), sum(loss_list) / len(X)))

    return sum(loss_list) / len(X)


def main():
    seed_everything(42)
    model = timm.models.create_model('resnet50', num_classes=10, pretrained=False)
    if opt.resume_path is not None:
        state_dict = torch.load(opt.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        del state_dict

    model = WrapModel(Normalize(opt.mean, opt.std), model=model).cuda()
    model.eval()
    X = ImageNette(root=opt.test_dir, transform=transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    if opt.adv_type == 'transfer':
        defender = attacks.TransferAttacker(
            radius=opt.max_epsilon,
            steps=50,
            step_size=2,
            norm_type= 'l-infty',
            ascending= not TARGETED,
            random_start=False,
            loss_type='logit' if TARGETED else 'ce',
            TI=False,
            DI=True,
            MI=True,
        )
    else:
        assert opt.adv_type == 'pgd'
        defender = attacks.PGDAttacker(
            radius=opt.max_epsilon,
            steps=50,
            step_size=2,
            norm_type='l-infty',
            ascending=not TARGETED,
            random_start=False,
        )
    for images, label, path in tqdm(data_loader):

        if TARGETED:
            gt = ((label+1) % 10).cuda()
        else:
            gt = label.cuda()
        images = images.cuda()
        adv_img = defender.perturb(model, torch.nn.functional.cross_entropy, images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, path, opt.output_dir)


if __name__ == '__main__':
    seed_everything(0)
    main()
    model_dict = {
        'resnet101': 'CIFAR_Train/imagenett-models-scratch/resnet101-seed-40/model-fin-model.pkl',
        'vgg16_bn': 'CIFAR_Train/imagenett-models-scratch/vgg16_bn-seed-40/model-fin-model.pkl',
        'densenet121':'CIFAR_Train/imagenett-models/densenet121-seed-40/model-fin-model.pkl',
        'mobilenetv2_100':'CIFAR_Train/imagenett-models/mobilenetv2-seed-40/model-fin-model.pkl',
        'xception': 'CIFAR_Train/imagenett-models-scratch/xception-seed-40/model-fin-model.pkl',
        'levit_128':'CIFAR_Train/imagenett-models/levit_128-seed-40/model-fin-model.pkl',
        'swin_base_patch4_window7_224': 'CIFAR_Train/imagenett-models/swin_base_patch4_window7_224-seed-40/model-fin-model.pkl',
        'vit_base_patch16_224': 'CIFAR_Train/imagenett-models/vit-seed-40/model-fin-model.pkl',
    }
    for key,val in model_dict.items():
        verify(key, val, opt.output_dir, opt.batch_size, on_imagenet=True)
        verify(key, val, opt.output_dir, opt.batch_size, on_imagenet=False)



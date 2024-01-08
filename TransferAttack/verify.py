"""Implementation of evaluate attack result."""
import os

import torch

from torchvision import transforms as T

from imagenet_csv import ImageNet
from CIFAR_Train.utils import get_arch
from torch.utils.data import DataLoader

batch_size = 50

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
# adv_dir = './dataset/outputs-vit-sam_TI-DI-MI-targeted'
adv_dir = './dataset/imagenet/outputs-1000-resnet50-targeted/'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

IMAGE_SIZE = 224


transforms = T.Compose(
    [T.Resize(IMAGE_SIZE), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def verify(model_name):
    model = get_arch(model_name, 'imagenet', pretrained=True).cuda()

    X = ImageNet(adv_dir, input_csv, transforms=transforms)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum_real = 0.
    sum_target = 0.
    loss_list = []

    for images, _, gt_real, gt_target in data_loader:
        gt_real = gt_real.cuda()
        gt_target = gt_target.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum_real += (torch.argmax(model(images), dim=1) != gt_real).detach().sum().cpu()
            sum_target += (torch.argmax(model(images), dim=1) == gt_target).detach().sum().cpu()
            loss_list.append(torch.nn.functional.cross_entropy(model(images), gt_real).item())
    print(model_name + '  Untargeted ASR = {:.2%}, Untargeted Loss = {:.2f}'.format(sum_real / len(X), sum(loss_list) / len(X)))
    print(model_name + '  Targeted ASR =  {:.2%}'.format(sum_target / len(X)))

    return sum(loss_list) / len(X)


def main():
    model_names = ['resnet50', 'resnet101', 'inception_v3', 'adv_inception_v3', 'ens_adv_inception_resnet_v2', 'densenet121', 'vgg16_bn',  'xception', 'swin_base_patch4_window7_224']
    print(adv_dir)
    total_loss = 0.
    for model_name in model_names:
        total_loss += verify(model_name)
        print("===================================================")
    print('Total Loss = {:.2f}'.format(total_loss) )


if __name__ == '__main__':
    main()


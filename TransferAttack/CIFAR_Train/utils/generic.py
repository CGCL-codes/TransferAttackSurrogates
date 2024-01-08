import pickle
import os
import random
import sys
import logging
import numpy as np
import pretrainedmodels
import timm
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from . import data
from . import imagenet_utils, losses, wasam, entropySGD


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt=1):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True):
    if dataset == 'imagenet' or dataset == 'imagenett':
        return imagenet_utils.get_transforms(dataset, train, is_tensor)

    if train:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4), ]
        elif dataset == 'GTSRB':
            comp1 = [transforms.Resize([32, 32])
                     ]
        elif dataset == 'tiny-imagenet':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8), ]
        elif dataset == 'cifar10-vit':
            comp1 = [
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),]
        else:
            raise NotImplementedError
    else:
        comp1 = []
        if dataset == 'cifar10-vit':
            comp1 = [
                transforms.Resize([224, 224]),
                transforms.RandomCrop(224)]
    if is_tensor:
        comp2 = []

    else:
        comp2 = [
            transforms.ToTensor(),
        ]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

    return trans


def get_filter(fitr):
    if fitr == 'averaging':
        return lambda x: cv2.blur(x, (3,3))
    elif fitr == 'gaussian':
        return lambda x: cv2.GaussianBlur(x, (3,3), 0)
    elif fitr == 'median':
        return lambda x: cv2.medianBlur(x, 3)
    elif fitr == 'bilateral':
        return lambda x: cv2.bilateralFilter(x, 9, 75, 75)

    raise ValueError


class WrapModel(torch.nn.Module):
    def __init__(self,  model, normalizer):
        super(WrapModel, self).__init__()
        self.model = model
        self.normal = normalizer

    def forward(self, input):
        return self.model(self.normal(input))
    def features(self, input):
        return self.model.features(self.normal(input))
    def eval(self):
        self.model.eval()



def get_normalize(dataset):
    class Normalize(torch.nn.Module):

        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.mean = mean
            self.std = std

        def forward(self, input):
            size = input.size()
            x = input.clone()
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
            return x

    if dataset == 'cifar10':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2471, 0.2435, 0.2616])
    elif dataset == 'tiny-imagenet':
        mean = torch.Tensor([0.4802, 0.4481, 0.3975])
        std = torch.Tensor([0.2770, 0.2691, 0.2821])
    elif dataset == 'cifar100':
        mean = torch.Tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        std = torch.Tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    elif dataset == 'cifar10-vit':
        mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        std = torch.Tensor([0.2471, 0.2435, 0.2616])
    elif dataset == 'imagenet' or dataset == 'imagenett' or dataset == 'imagenet-10':
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
    elif dataset == 'GTSRB':
        mean = torch.Tensor([0, 0, 0])
        std = torch.Tensor([1, 1, 1])
    else:
        raise NotImplementedError

    return Normalize(mean=mean, std=std)


def get_dataset(dataset, root='./data', train=True, fitr=None, use_cutout=False, fool=False, cutout_size=16):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_dataset(dataset, root, train)

    transform = get_transforms(dataset, train=train, is_tensor=False) if not fool else get_transforms(dataset,train=False,is_tensor=False)
    if use_cutout:
        print(cutout_size)
        transform.transforms.append(data.Cutout(cutout_size))
    lp_fitr  = None if fitr is None else get_filter(fitr)

    if dataset == 'cifar10' or dataset =='cifar10-vit':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'tiny-imagenet':
        target_set = data.datasetTinyImageNet(root=root, train=train, transform=transform)
        x, y = target_set.x, target_set.y
    elif dataset == 'GTSRB':
        target_set = data.datasetGTSRB(root=root, train=train, transform=transform)
        return target_set

    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform, lp_fitr)


def get_indexed_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)

    if train:
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    else:
        # target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_indexed_tensor_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_tensor_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_clean_loader(dataset, batch_size, root='./data', train=True, cutout_size=16,use_cutmix=False, use_mixup=False, use_cutout=False, use_same_mixup=False, mixup_prob=0.5, cutmix_prob=0.5, index=False, fool=False):
    target_set = get_dataset(dataset, root=root, train=train,use_cutout=use_cutout, cutout_size=cutout_size, fool=fool)
    target_set = data.Dataset(x=target_set.x, y=np.array(target_set.y), transform=target_set.transform, fitr=target_set.fitr)

    if use_cutmix:
        target_set = data.CutMix(dataset=target_set, num_class=10, prob=cutmix_prob)
    elif use_mixup:
        target_set = data.MixUp(dataset=target_set, num_class=10, prob=mixup_prob)
    elif use_same_mixup:
        target_set = data.SameMixUp(dataset=target_set, num_class=10, prob=mixup_prob)
    if index :
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0,  fitr=None, use_cutmix=False, use_mixup=False, use_cutout=False,fool=False,index=False):

    if dataset == 'imagenet' or dataset == 'imagenett':
        return imagenet_utils.get_poisoned_loader(
                dataset, batch_size, root, train, noise_path, noise_rate,  fitr)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr,use_cutout=use_cutout,fool=fool)

    if noise_path is not None and 'l2' not in noise_path:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)
        assert isinstance(raw_noise, np.ndarray)
        shape_0 = raw_noise.shape[0]
        if shape_0 != len(target_set):
            uni_noise = raw_noise.copy()
            raw_noise = np.expand_dims(np.zeros(raw_noise.shape[1:]), axis=0)
            raw_noise = np.repeat(raw_noise, len(target_set), axis=0)
            for id, y in enumerate(target_set.y):
                raw_noise[id] = uni_noise[int(y)]
            raw_noise = raw_noise.astype(np.int8)
            # raw_noise = np.repeat(np.expand_dims(raw_noise, axis=1), len(target_set)//shape_0, axis=1)
            # raw_noise = raw_noise.reshape([-1, raw_noise.shape[2], raw_noise.shape[3], raw_noise.shape[4]])
        assert raw_noise.dtype == np.int8
        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)
        noise += raw_noise

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = target_set.x.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.x = imgs
    else:
        assert 'l2' in noise_path
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)
        assert isinstance(raw_noise, np.ndarray)

        noise = np.zeros_like(raw_noise)
        noise += raw_noise

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0, 2, 3, 1])
        target_set.x = noise.clip(0, 255).astype(np.uint8)

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if use_cutmix:
        target_set = data.CutMix(dataset=target_set, num_class=10)
    elif use_mixup:
        target_set = data.MixUp(dataset=target_set, num_class=10)

    if index :
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_dataset(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0,  fitr=None, use_cutmix=False, use_mixup=False, use_cutout=False,fool=False,index=False):

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr,use_cutout=use_cutout,fool=fool)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)
        assert isinstance(raw_noise, np.ndarray)
        shape_0 = raw_noise.shape[0]
        if shape_0 != len(target_set):
            uni_noise = raw_noise.copy()
            raw_noise = np.expand_dims(np.zeros(raw_noise.shape[1:]), axis=0)
            raw_noise = np.repeat(raw_noise, len(target_set), axis=0)
            for id, y in enumerate(target_set.y):
                raw_noise[id] = uni_noise[int(y)]
            raw_noise = raw_noise.astype(np.int8)

        assert raw_noise.dtype == np.int8
        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)

        noise += raw_noise

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = target_set.x.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.x = imgs

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    return target_set

def get_clear_loader(
        dataset, batch_size, root='./data', train=True,
        noise_rate=1.0, poisoned_indices_path=None, fitr=None):

    if dataset == 'imagenet' or dataset == 'imagenett':
        return imagenet_utils.get_clear_loader(
                dataset, batch_size, root, train, noise_rate, poisoned_indices_path)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr)
    data_nums = len(target_set)

    if poisoned_indices_path is not None:
        with open(poisoned_indices_path, 'rb') as f:
            poi_indices = pickle.load(f)
        indices = np.array( list( set(range(data_nums)) - set(poi_indices) ) )

    else:
        indices = np.random.permutation(range(data_nums))[: int( data_nums * (1-noise_rate) )]

    ''' select clear examples '''
    target_set.x = target_set.x[indices]
    target_set.y = np.array(target_set.y)[indices]

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


# TODO: 1. for datasets like cifar10, cifar100, I use files \  2. imagenett and imagenet, I use timm, imagemett use checkpoints , imagenet uses timm pretrained  models

# def get_arch(arch, dataset, model_type=None):
#     if model_type == 'torchvision':
#         return pretrainedmodels.__dict__[arch]()
#
#     if dataset == 'cifar10':
#         in_dims, out_dims = 3, 10
#         model_type = 'defined'
#     elif dataset == 'cifar100':
#         in_dims, out_dims = 3, 100
#         model_type = 'defined'
#     elif dataset == 'imagenet':
#          in_dims, out_dims = 3, 1000
#          model_type = 'timm'
#     elif dataset == 'imagenett':
#         in_dims, out_dims = 3, 10
#         model_type = 'timm'
#     else:
#         raise NotImplementedError('dataset {} is not supported'.format(dataset))
#     if model_type == 'defined':
#         if arch == 'resnet18':
#             return models.resnet18(in_dims, out_dims)
#         elif arch == 'resnet50':
#             return models.resnet50(in_dims, out_dims)
#         elif arch == 'resnet20':
#             return models.resnet20(out_dims)
#         elif arch == 'resnet56':
#             return models.resnet56(out_dims)
#         elif arch == 'resnet32':
#             return models.resnet32(out_dims)
#         elif arch == 'resnet44':
#             return models.resnet44(out_dims)
#         elif arch == 'wrn-34-10':
#             return models.wrn34_10(in_dims, out_dims)
#         elif arch == 'vgg11-bn':
#             return models.vgg11_bn(in_dims, out_dims)
#         elif arch == 'vgg16-bn':
#             return models.vgg16_bn(in_dims, out_dims)
#         elif arch == 'vgg19-bn':
#             return models.vgg19_bn(in_dims, out_dims)
#         elif arch == 'densenet-121':
#             return models.densenet121(num_classes=out_dims)
#         elif arch == 'vit_patch_2':
#             return models.ViT(
#             image_size=36,
#             patch_size=2,
#             num_classes=out_dims,
#             dim=512,
#             depth=6,
#             heads=8,
#             mlp_dim=512,
#         )
#         elif arch == 'vit_patch_4':
#             return models.ViT(
#             image_size=36,
#             patch_size=4,
#             num_classes=out_dims,
#             dim=512,
#             depth=6,
#             heads=8,
#             mlp_dim=512,
#             dropout=0.1,
#             emb_dropout=0.1
#         )
#
#         else:
#             raise NotImplementedError('architecture {} is not supported'.format(arch))
#     elif model_type == 'timm':
#          return timm.create_model(model_name=arch, num_classes=out_dims,  pretrained=True)


def get_arch(arch, dataset, pretrained=False, resume=False):
    if dataset == 'imagenet':
        return timm.create_model(model_name=arch, num_classes=1000, pretrained=pretrained)
    elif dataset == 'imagenett' or dataset == 'imagenet-10':
        return timm.create_model(model_name=arch, num_classes=10, pretrained=pretrained)
    elif dataset=='tiny-imagenet':
        return timm.create_model(model_name=arch, num_classes=200, pretrained=pretrained)
    elif dataset == 'cifar10' or dataset=='cifar100' or dataset == 'GTSRB':
        if dataset == 'cifar10':
            num_class = 10
        elif dataset == 'cifar100':
            num_class = 100
        else:
            num_class = 43
        if not resume:
            import models
        else:
            from .. import models
        return getattr(models, arch)(num_class)

    elif dataset == 'cifar10-vit':
        if arch == 'ViT':
            model = timm.create_model(model_name='vit_base_patch16_224',  pretrained=pretrained)
            model.head = torch.nn.Linear(model.head.in_features, 10)
            return model
        elif arch == 'ViT-sam':
            model = timm.create_model(model_name='vit_base_patch16_224_sam', pretrained=pretrained)
            model.head = torch.nn.Linear(model.head.in_features, 10)
            return model
    else:
        raise NotImplementedError('architecture {} is not supported in dataset'.format(arch, dataset))


def get_optim(optim, model,args, lr=0.1, weight_decay=1e-4, momentum=0.9, rho=0.05, sam=False, swa=False):
    if optim == 'sgd':
        if swa:
            return wasam.WASAM(model.parameters(), torch.optim.SGD, lr=lr, weight_decay=weight_decay, momentum=momentum, rho=rho)
        if not sam:
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        print("using sam")
        return losses.SAM(model.parameters(), torch.optim.SGD, lr=lr, weight_decay=weight_decay, momentum=momentum, rho=rho)
    elif optim == 'adam':
        if not sam:
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return losses.SAM(model.parameters(), torch.optim.Adam, lr=lr, weight_decay=weight_decay, rho=rho)
    elif optim == 'adamw':
        if not sam:
            return torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
        return losses.SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay, rho=rho)
    elif optim == 'sgd-lsam':
        return losses.LSAM(model.parameters(), torch.optim.SGD, lr=lr, weight_decay=weight_decay, momentum=momentum, rho=rho)
    elif optim == 'adam-lsam':
        return losses.LSAM(model.parameters(), torch.optim.Adam, lr=lr, weight_decay=weight_decay, rho=rho)
    elif optim == 'adamw-lsam':
        return losses.LSAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay, rho=rho)
    elif optim == 'entropySGD':
        return entropySGD.EntropySGD(model.parameters(), lr=lr, L=args.L, weight_decay=weight_decay,momentum=momentum,gamma_fix=args.gamma_fix)
    elif optim == 'APM':
        return losses.APM(model, torch.optim.SGD, APM_gamma=args.gamma_fix, times=args.times, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    fh = '{}/{}_log.txt'.format(args.save_dir, args.save_name)

    logging.basicConfig(filename=fh,filemode='a', level=logging.INFO, format=fmt)
    logger = logging.getLogger()
    return logger


def evaluate(model, criterion, loader, cpu):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()

    pbar = tqdm(loader, total=len(loader), colour='green',position=2, leave=False)

    for x, y in pbar:
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))
        pbar.set_description("Test: acc %.2f Loss: %.2f" % (acc.average() * 100, loss.average() ))

    return acc.average(), loss.average()


def robust_evaluate(model, criterion,  loader,  attacker, cpu):
    acc = AverageMeter()
    loss = AverageMeter()
    model.eval()
    pbar = tqdm(loader, total=len(loader), colour='green',position=2, leave=False)
    for x, y in pbar:
        if not cpu: x, y = x.cuda(), y.cuda()
        x = attacker(x, y)
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))
        pbar.set_description("Robust Test: acc %.2f Loss: %.2f" % (acc.average() * 100, loss.average()))
    return acc.average(), loss.average()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

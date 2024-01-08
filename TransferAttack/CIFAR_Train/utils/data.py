import random

from PIL import Image
import numpy as np
import torch
import torchvision
import os
import pickle
from torch.utils.data.dataset import Dataset as TDataset

class ElementWiseTransform():
    def __init__(self, trans=None):
        self.trans = trans

    def __call__(self, x):
        if self.trans is None: return x
        return torch.cat( [self.trans( xx.view(1, *xx.shape) ) for xx in x] )


class IndexedTensorDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        ''' transform HWC pic to CWH pic '''
        x = torch.tensor(x, dtype=torch.float32).permute(2,0,1)
        return x, y, idx

    def __len__(self):
        return len(self.x)


class Dataset():
    def __init__(self, x, y, transform=None, fitr=None):
        self.x = x
        self.y = y
        self.transform = transform
        self.fitr = fitr

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        ''' low pass filtering '''
        if self.fitr is not None:
            x = self.fitr(x)

        ''' data augmentation '''
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )

        return x, y

    def __len__(self):
        return len(self.x)


class IndexedDataset():
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.ii = np.array( range(len(x)), dtype=np.int64 )
        self.transform = transform

    def __getitem__(self, idx):
        x, y, ii = Image.fromarray(self.x[idx]), self.y[idx], self.ii[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, ii

    def __len__(self):
        return len(self.x)


def datasetCIFAR10(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR10(root=root, train=train,
                        transform=transform, download=True)

def datasetCIFAR100(root='./path', train=True, transform=None):
    return torchvision.datasets.CIFAR100(root=root, train=train,
                        transform=transform, download=True)

def datasetGTSRB(root='./path', train=True, transform=None):
    if train:
        root = os.path.join(root, 'GTSRB_train.pkl')
    else:
        root = os.path.join(root, 'GTSRB_val.pkl')
    with open(root, 'rb') as f:
        dat = pickle.load(f)
    return Dataset(dat['data'], dat['targets'], transform)

def datasetTinyImageNet(root='./path', train=True, transform=None):
    if train: root = os.path.join(root, 'tiny-imagenet_train.pkl')
    else : root = os.path.join(root, 'tiny-imagenet_val.pkl')
    with open(root, 'rb') as f:
        dat = pickle.load(f)
    return Dataset(dat['data'], dat['targets'], transform)

    # root = os.path.join(root, 'tiny-imagenet-200')
    # if train: root = os.path.join(root, 'train')
    # else: root = os.path.join(root, 'val', 'images')
    # raw_dataset = torchvision.datasets.ImageFolder(root)
    # xx, yy = [], []
    # for i in range( len(raw_dataset) ):
    #     x, y = raw_dataset[i]
    #     x = np.array(x)
    #     xx.append( x.reshape(1, *x.shape) )
    #     yy.append( y )
    # xx = np.concatenate(xx)
    # yy = np.array(yy)
    #
    # dat = {'data':xx, 'targets':yy}
    # if train: save_name = 'tiny-imagenet_train.pkl'
    # else: save_name = 'tiny-imagenet_val.pkl'
    #
    # import pickle
    # with open('./data/{}'.format(save_name), 'wb') as f:
    #     pickle.dump(dat, f)
    # exit()
    # return Dataset(xx, yy, transform)


class Loader():
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, num_workers=2):
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixUp(TDataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class SameMixUp(TDataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.label_indices = np.argsort(self.dataset.y)

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self) // self.num_class))

            img2, lb2 = self.dataset[self.label_indices[len(self) // self.num_class * int(lb) + rand_index]]

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)



class CutMix(TDataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

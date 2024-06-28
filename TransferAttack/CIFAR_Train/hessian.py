import random

import torchvision.datasets
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Any
from torchvision.datasets.folder import default_loader, Optional, Callable

DEFAULT_PHYS_BS = 100

random.seed(12)


def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])


class TakeSomeImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            num: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(TakeSomeImageFolder, self).__init__(root, transform=transform,
                                                  target_transform=target_transform,
                                                  loader=loader,
                                                  is_valid_file=is_valid_file)
        self.num_to_keep = num
        self.targets = self.targets
        index_list = random.sample(range(len(self.targets)), num)
        self.samples = tuple([self.samples[i] for i in index_list])
        self.targets = tuple([self.targets[i] for i in index_list])

    def __len__(self):
        return self.num_to_keep


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()


def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, physical_batch_size: int = DEFAULT_PHYS_BS):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int, which='LA'):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs, which=which)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, physical_batch_size=1000, which='LA'):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs, which=which)

    return evals


def get_cosine_similarity(model1, model2, X, y, criterion):
    X.requires_grad_(True)
    grad1 = torch.autograd.grad(criterion(model1(X), y), X)[0]
    grad1 = grad1.view(grad1.shape[0], -1)
    grad2 = torch.autograd.grad(criterion(model2(X), y), X)[0]
    grad2 = grad2.view(grad2.shape[0], -1)
    norm1 = grad1.norm(dim=1)
    norm2 = grad2.norm(dim=1)
    cos = [torch.dot(grad1[i] / norm1[i], grad2[i] / norm2[i]) for i in range(grad1.shape[0])]
    return sum(cos)


def get_cosine_similarity_imagenette(model1, model2, X, y, criterion):
    X2 = X.clone().detach()
    X2.requires_grad_(True)
    X.requires_grad_(True)
    grad1 = torch.autograd.grad(criterion(model1(X), y), X)[0]
    grad1 = grad1.view(grad1.shape[0], -1)
    grad2 = torch.autograd.grad(criterion(model2(X2),y), X2)[0]
    grad2 = grad2.view(grad2.shape[0], -1)
    cos = torch.cosine_similarity(grad1, grad2)
    return sum(cos)

def get_cosine_similarity_imagenette_logit(model1, model2, X, y, class_num=10):
    assert X.shape[0] == 1
    cos = 0

    X2 = X.clone().detach()


    X.requires_grad_(True)
    logit1 = model1(X)[0][y]
    grad1 = torch.autograd.grad(logit1, X)
    grad1 = X.grad

    X2.requires_grad_(True)
    logit2 = model2(X2)[0][y]
    grad2 = torch.autograd.grad(logit2, X2)[0]
    grad2 = X2.grad
    
    grad1 = grad1.view(grad1.shape[0], -1)
    grad2 = grad2.view(grad2.shape[0], -1)
    print(grad1)
    
    print(torch.cosine_similarity(grad1, grad2))
    
    return   sum(torch.cosine_similarity(grad1, grad2))

        
        

def get_input_l1_norm(model, X, y, criterion):
    X.requires_grad_(True)
    grad = torch.autograd.grad(criterion(model(X), y), X)[0]
    return torch.norm(grad, p=1, dim=0)


def get_input_l0_norm(model, X, y, criterion):
    X.requires_grad_(True)
    grad = torch.autograd.grad(criterion(model(X), y), X)[0]
    return torch.norm(grad, p=0, dim=0)


def get_jac_mine(model, x):
    from torch.nn.utils import parameters_to_vector
    grads = torch.autograd.grad(list(model(x).view(-1)), inputs=x, create_graph=True)
    grads = parameters_to_vector(grads)
    return grads


def compute_hvp_sample(network: nn.Module, loss_fn: nn.Module,
                 vector: Tensor, X: Tensor, y: Tensor, nparams : int):
    X = X.cuda()
    y = y.cuda()
    X.requires_grad_(True)
    for pp in network.parameters():
        pp.requires_grad = False
    hvp = torch.zeros(nparams, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    loss = loss_fn(network(X), y)
    grads = torch.autograd.grad(loss, inputs=X, create_graph=True)
    grads = torch.cat(grads).view(-1)
    dot = grads.mul(vector).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, X, retain_graph=True)]
    hvp += torch.cat(grads).view(-1)
    return hvp


def compute_multi_db_distance_for_single_sample(network, X, y):
    for pp in network.parameters():
        pp.requires_grad = False
    assert X.shape[0] == 1
    logits = network(X).view(-1)
    ndim = logits.shape[-1]
    y = int(y.item())
    grads = torch.autograd.functional.jacobian(lambda x: torch.nn.Softmax()(network(x)), X)
    grads = grads.view(ndim, -1)
    db_list = []
    for i in range(ndim):
        if i == y:
            db_list.append(0)
        else:
            db_list.append((torch.abs(logits[i] - logits[y]) / torch.norm(grads[i] - grads[y])).item())
    return db_list


def compute_multi_db_curvature(network, X, y):
    for pp in network.parameters():
        pp.requires_grad = False
    assert X.shape[0] == 1
    logits = network(X).view(-1)
    ndim = logits.shape[-1]
    y = int(y.item())
    torch.nn.Softmax()
    grads = torch.autograd.functional.jacobian(lambda x: torch.nn.Softmax()(network(x)), X)
    grads = grads.view(ndim, -1)
    # we use jacobian to approximate hessian， see ECCV2018
    db_curvature = []
    for i in range(ndim):
        if i == y:
            db_curvature.append(0)
        else:
            db_curvature.append(
                torch.abs(torch.pow(torch.dot(grads[i].view(-1), X.view(-1)), 2) - torch.pow(torch.dot(grads[y].view(-1), X.view(-1)), 2)).item()
            )
    return db_curvature


def get_hessian_eigenvalues_from_sample(network: nn.Module, loss_fn: nn.Module, sample: Tensor, label: Tensor,
                                         neigs=6, which='LA'):
    """ Compute the leading Hessian eigenvalues for per sample. """
    ndim = sample.view(-1).shape[0]
    hvp_delta = lambda delta: compute_hvp_sample(network, loss_fn, delta, sample, label, ndim).detach().cpu()
    evals, evecs = lanczos(hvp_delta, ndim, neigs=neigs, which=which)
    return evals


# def

if __name__ == '__main__':

    from torchvision import transforms

    transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    IMAGENET_TRAIN = '../../DATASET/train/clean/ILSVRC2012_img_train'

    model_path_list = ['../save_models/RobustModels/resnet50_l2_eps0.ckpt',
                       '../save_models/RobustModels/resnet50_l2_eps0.01.ckpt',
                        '../save_models/RobustModels/resnet50_l2_eps0.03.ckpt',
                        '../save_models/RobustModels/resnet50_l2_eps0.05.ckpt',
                        '../save_models/RobustModels/resnet50_l2_eps0.1.ckpt',
                         '../save_models/RobustModels/resnet50_l2_eps0.25.ckpt',
                        '../save_models/RobustModels/resnet50_l2_eps0.5.ckpt',
                       '../save_models/RobustModels/resnet50_l2_eps1.ckpt',
                       '../save_models/RobustModels/resnet50_l2_eps3.ckpt',
                       '../save_models/RobustModels/resnet50_l2_eps5.ckpt',
                      ]

    dataset = TakeSomeImageFolder(root=IMAGENET_TRAIN, num=5000, transform=transformer)

    from robustness import model_utils, datasets

    ds = datasets.ImageNet('../../DATASET/train/clean/ILSVRC2012_img_train')
    sharpness_list = []
    for model_path in tqdm(model_path_list):
        model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
                                                      resume_path=model_path, pytorch_pretrained=False)
        model = model.model
        sharpness = get_hessian_eigenvalues(model, torch.nn.CrossEntropyLoss(reduction='sum'), dataset, neigs=1,
                                            physical_batch_size=50)
        print(sharpness)
        sharpness_list.append(sharpness[0].item())
    print(sharpness_list)

import argparse


def add_shared_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'vgg11-bn', 'vgg16-bn', 'vgg19-bn', 'densenet-121', 'inception-resnet-v1', 'inception-v3', 'wrn-34-10', 'vit_patch_2', 'vit_patch_4'],
                        help='choose the model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenett'],
                        help='choose the dataset')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size')

    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='select which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='set the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='set the weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum for SGD')

    parser.add_argument('--parallel', action='store_true',
                        help='select to use distributed data parallel')
    parser.add_argument('--cpu', action='store_true',
                        help='select to use cpu, otherwise use gpu')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='set the path to the exp data')
    parser.add_argument('--save-dir', type=str, default='./temp',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default='temp-name',
                        help='set the save name of the experiment result')

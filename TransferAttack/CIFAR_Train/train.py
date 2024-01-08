import math
import pickle
import argparse
import os
from torch.utils.data import DataLoader

from utils import data

import timm

import torch
import wandb
from tqdm import tqdm
import torchattacks
import utils
from utils import LabelSmoothingLoss, AverageMeter,  seed_everything, WrapModel
from utils.losses import CutMixCrossEntropyLoss
from torch.nn.utils import parameters_to_vector
from trade_loss import trades_loss
from utils.imagenet_utils import get_dataset
from utils import WASAM, MultipleSWAModels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ResNet18',
                        help='choose the model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cifar10-vit', 'imagenett', 'tiny-imagenet', 'GTSRB', 'imagenet-10'],
                        help='choose the dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size')
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw','entropySGD' , 'APM'],
                        help='select which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.1, help='set the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='set the weight decay rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='set the momentum for SGD')
    parser.add_argument('--cpu', action='store_true',
                        help='select to use cpu, otherwise use gpu')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='set the path to the exp data')
    parser.add_argument('--save-dir', type=str, default='./temp',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default='temp-name',
                        help='set the save name of the experiment result')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')
    parser.add_argument('--label-smoothing', action='store_true')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--cutmix-prob', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--warm-up', action='store_true')
    parser.add_argument('--warm-up-epoch', type=int, default=10)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--sam', action='store_true')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa-start-coeff', type=float, default=0.75)
    parser.add_argument('--remain-constant', action='store_true')
    parser.add_argument('--constantLR', type=float, default=0.05)
    parser.add_argument('--rho', type=float, default=0.05, help='sam parameters')
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--reg-type', type=str, default='jr', choices=['jr', 'ig', 'jr_true',  'model', 'logits_model'])
    parser.add_argument('--jr-beta', type=float, default=1.)
    parser.add_argument('--ig-beta', type=float, default=100.)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--samemixup', action='store_true')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-size', type=int, default=32)
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--TRADE', action='store_true')
    parser.add_argument('--TRADE-beta', type=float, default=0.5)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--scale-factor', type=float, default=1.5)
    parser.add_argument('--pgd-radius', type=float, default=0,
                        help='set the perturbation radius in pgd')
    parser.add_argument('--pgd-steps', type=int, default=0,
                        help='set the number of iteration steps in pgd')
    parser.add_argument('--pgd-step-size', type=float, default=0,
                        help='set the step size in pgd')
    parser.add_argument('--pgd-random-start', action='store_true',
                        help='if select, randomly choose starting points each time performing pgd')
    parser.add_argument('--pgd-norm-type', type=str, default='l-infty',
                        choices=['l-infty', 'l2', 'l1'],
                        help='set the type of metric norm in pgd')
    # parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--times', type=int, default=20)
    parser.add_argument('--gamma-fix', type=float, default=0)
    parser.add_argument('--constant', action='store_true')
    parser.add_argument('--wsam', action='store_true')
    parser.add_argument('--setting-lr', action='store_true')
    return parser.parse_args()


def save_checkpoint(save_dir, save_name, model, optim, log):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'model_state_dict': utils.get_model_state(model),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f)


def get_warm_up_with_cosine_lr(args, optim):
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epoch  if epoch <= args.warm_up_epoch else 0.5 * (
            math.cos((epoch - args.warm_up_epoch) / (args.epoch - args.warm_up_epoch) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warm_up_with_cosine_lr)
    return scheduler


def get_jac_loss_mine(logits, x):
    from torch.nn.utils import parameters_to_vector
    grads = torch.autograd.grad(list(logits.view(-1)), inputs=x, create_graph=True)
    grads = parameters_to_vector(grads)
    return torch.norm(grads)

def get_jac_loss_true(model, X):
    from torch.autograd.functional import jacobian
    grads = jacobian(lambda x_: model(x_), X)
    return torch.norm(grads[:,:,0].view(-1))


def get_jac_loss(logits, x):
    from jacobian import JacobianReg
    reg = JacobianReg()
    return reg(x, logits)

def get_input_gradient_loss(loss, x):
    grads = torch.autograd.grad(loss, inputs=x, create_graph=True)
    grads = parameters_to_vector(grads)
    return torch.norm(grads)


def get_model_regularization(loss, model, gamma):
    grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    grads = parameters_to_vector(grads)
    return (gamma / 2) * torch.sum(grads ** 2)


def model_regularization_DB(loss, model, gamma):
    loss.backward(create_graph=True)
    loss_DB = (gamma / 2) * sum([torch.sum(p.grad ** 2) for p in model.parameters()])
    return loss_DB


def get_logits_model_regularization(logits, model):
    grads = torch.autograd.grad(list(logits.view(-1)), inputs=model.parameters(), create_graph=True)
    grads = parameters_to_vector(grads)
    return torch.norm(grads)


def train(model, normalize, optim, criterion, train_loader, test_loader1):
    log = dict()
    if args.swa:
        swa_start_epoch = int(args.epoch * args.swa_start_coeff)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch) if not args.warm_up else get_warm_up_with_cosine_lr(args, optim)
    if args.swa and args.remain_constant:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=swa_start_epoch)
    if args.constant:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.00001)
    if args.setting_lr:
        scheduler = None
    best_acc = 0
    wrap_model = WrapModel(model=model, normalizer=normalize)
    if args.parallel:
        wrap_model = torch.nn.DataParallel(wrap_model)
    if args.robust:
        best_robust_acc = 0
        if args.pgd_norm_type == 'l2':
            attack = torchattacks.PGDL2
            attacker = attack(model=wrap_model, eps=args.pgd_radius, alpha=args.pgd_step_size, steps=args.pgd_steps,
                              random_start=args.pgd_random_start)
        elif args.pgd_norm_type == 'l-infty':
            attack = torchattacks.PGD
            attacker = attack(model=wrap_model, eps=args.pgd_radius / 255, alpha=args.pgd_step_size / 255, steps=args.pgd_steps,
                              random_start=args.pgd_random_start)
        else:
            raise NotImplementedError


    for epoch in tqdm(range(args.epoch)):
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, total=len(train_loader))

        if args.setting_lr:
            stage = epoch // 10 + 1
            current_epoch = epoch % 10 + 1
            # 计算学习率
            if current_epoch <= 8:
                lr = 1.0 / stage
            else:
                lr = 10 ** (-1 - stage / 2)
            # 设置当前epoch的学习率
            # 设置当前epoch的学习率
            for param_group in optim.param_groups:
                param_group['lr'] = lr

        for x, y in pbar:
            if not args.cpu:
                x, y = x.cuda(), y.cuda()
            model.zero_grad()
            optim.zero_grad()
            if args.robust and not args.TRADE:
                model.eval()
                x = attacker(x, y)

            model.train()

            if not args.reg:
                if args.TRADE:
                    def closure():
                        loss, logits = trades_loss(model=wrap_model,
                                           x_natural=x,
                                           y=y,
                                           optimizer=optim,
                                           step_size=args.pgd_step_size,
                                           epsilon=args.pgd_radius,
                                           perturb_steps=args.pgd_steps,
                                           distance=args.pgd_norm_type)
                        loss.backward()
                        return loss, logits

                else:
                    def closure():
                        logits = wrap_model(x)
                        loss = criterion(logits, y).mean() if args.wsam else criterion(logits, y)
                        loss.backward()
                        return loss, logits

                if args.wsam:
                    def ascent_closure():
                        logits = wrap_model(x)

                        loss = criterion(logits, y)
                        sorted_indices = torch.argsort(loss)
                        selected_indices = sorted_indices[:int(2 / 3 * len(loss))]
                        selected_loss_mean = torch.mean(loss[selected_indices])
                        selected_loss_mean.backward()
                        return loss.mean(), logits
            else:
                x.requires_grad = True
                def closure():
                    logits = wrap_model(x)
                    loss = criterion(logits, y)
                    # if args.reg_type == 'jr_true':
                    #     loss += args.jr_beta * get_jac_loss_true(wrap_model, x)
                    if args.reg_type == 'jr':
                        loss += args.jr_beta * get_jac_loss(logits, x)
                        loss /= x.shape[0]
                    # elif args.reg_type == 'mine':
                    #     loss += args.jr_beta * get_jac_loss_mine(logits, x)
                    elif args.reg_type == 'model':
                        loss += get_model_regularization(loss, model, args.jr_beta)
                    # elif args.reg_type == 'logits_model':
                    #     loss += args.jr_beta * get_logits_model_regularization(logits, model)
                    else:
                        assert args.reg_type == 'ig'
                        loss += args.ig_beta * get_input_gradient_loss(loss, x)
                        loss /= x.shape[0]
                    loss.backward()
                    return loss, logits

            if args.wsam:
                loss, logits= ascent_closure()
            else:
                loss, logits = closure()

            if (not args.sam) and (not args.entropy) and (args.optim != 'APM'):
                if not args.swa:
                    optim.step()
                else:
                    optim.base_optimizer.step()
            else:

                optim.step(closure=closure)

            _, predicted = torch.max(logits.data, 1)

            if isinstance(criterion,utils.losses.CutMixCrossEntropyLoss):
                y = torch.max(y.data, 1)[1]

            acc = (predicted == y).sum().item() / y.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Train acc %.2f Loss: %.2f" % (acc_meter.mean * 100, loss_meter.mean))
            utils.add_log(log, 'acc', acc)
            utils.add_log(log, 'loss', loss.item())
        if not args.setting_lr:
            scheduler.step()
        if args.swa:
            if epoch >= swa_start_epoch:
                optim.update_swa()
            # before model evaluation, swap weights with averaged weights
                optim.swap_swa_sgd()
                optim.bn_update(train_loader, wrap_model, device="cuda:0")  # <-------------- Update batchnorm statistics

            # if remain constant and reach to swa start, set lr to be constant
            if args.remain_constant and epoch >= swa_start_epoch -1:
                optim.param_groups[0]['lr'] = args.constantLR

        test_acc, test_loss = utils.evaluate(wrap_model, torch.nn.CrossEntropyLoss(), test_loader1, args.cpu)
        if args.robust:
            robust_acc, robust_loss = utils.robust_evaluate(wrap_model, criterion, test_loader1, attacker, args.cpu)
            wandb.log({'RobustAcc/train': acc_meter.mean, 'RobustLoss/train': loss_meter.mean, 'CleanAcc/val': test_acc,
                       'CleanLoss': test_loss, 'RobustAcc/val': robust_acc, 'RobustLoss/val': robust_loss})

            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_checkpoint(args.save_dir, 'best-robust-eval', model, optim, log)
        else:
            wandb.log({'train acc': acc_meter.mean, 'train loss': loss_meter.mean, 'test acc': test_acc, 'test loss': test_loss})
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(args.save_dir, 'best-clean-eval', model, optim, log)
        save_checkpoint(args.save_dir, 'model-fin', model, optim, log)
        if args.swa:
            if epoch >= swa_start_epoch: 
                optim.swap_swa_sgd()
                save_checkpoint(args.save_dir, 'model-fin-sgd', model, optim, log)

    return model, optim, log


def get_image_loader(args):
    train_dataset = get_dataset(root=args.data_dir,dataset=args.dataset,train=True)
    if args.dataset == 'imagenett' or args.dataset == 'imagenet-10':
        num_classes = 10
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        raise NotImplementedError
    if args.mixup:
        train_dataset = data.MixUp(dataset=train_dataset, num_class=num_classes,prob=args.mixup_prob)
    if args.cutmix:
        train_dataset = data.CutMix(dataset=train_dataset, num_class=num_classes, prob=args.cutmix_prob)
    if args.cutout:
        train_dataset.transform.transforms.append(data.Cutout(args.cutout_size))

    test_dataset = get_dataset(root=args.data_dir,dataset=args.dataset,train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
    return train_loader, test_loader


def main(args):
    model = utils.get_arch(args.arch, args.dataset, pretrained=args.finetune)
    normalize = utils.get_normalize(args.dataset)
    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        del state_dict

    if not args.label_smoothing:
        criterion = CutMixCrossEntropyLoss if args.cutmix or args.mixup or args.samemixup else torch.nn.CrossEntropyLoss
        if args.reg:
            if args.reg_type == 'model':
                criterion = criterion()
            else:
                criterion = criterion(reduction='sum')
        else:
            if args.wsam:
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
            else:
                criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothingLoss(classes=100 if args.dataset == 'cifar100' else 10, smoothing=args.smoothing)

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    optim = utils.get_optim(
        args.optim, model, args,
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,sam=args.sam, rho=args.rho, swa=args.swa)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset=='GTSRB':
        train_loader = utils.get_clean_loader(dataset=args.dataset, batch_size=args.batch_size,cutout_size=args.cutout_size, root=args.data_dir,train=True,use_mixup=args.mixup,use_same_mixup=args.samemixup, use_cutmix=args.cutmix,use_cutout=args.cutout,cutmix_prob=args.cutmix_prob, mixup_prob=args.mixup_prob)
        test_loader = utils.get_clean_loader(dataset=args.dataset, batch_size=args.batch_size,root=args.data_dir,train=False)
    else:
        assert args.dataset == 'imagenett' or args.dataset == 'tiny-imagenet' or args.dataset == 'imagenet-10'
        train_loader, test_loader = get_image_loader(args)
    model, optim, log = train(model, normalize, optim, criterion, train_loader, test_loader)

    save_checkpoint(args.save_dir, 'model-fin', model, optim, log)


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed) # 12 for target model
    if args.dataset == 'cifar10':
        project_name ='TransferAttack-ModelTrainRobust'
    elif args.dataset == 'imagenett':
        if 'scratch' in args.save_dir:
            project_name = 'TransferAttack-Imagenette-Scratch'
        else:
            project_name ='TransferAttack-Imagenett'

    elif args.dataset == 'GTSRB':
        project_name = 'TransferAttack-GTSRB'
    elif args.dataset == 'imagenet-10':
        project_name = 'TransferAttack-imagenet10'
    else:
        print(args.dataset)
        assert args.dataset == 'tiny-imagenet'
        project_name ='TransferAttack-tiny'

    try:
        wandb.init(project=project_name, name=args.save_dir.split('/')[-1], entity='aisp2020')
        wandb.config.update(args)
        main(args)
    except Exception as e:
        print(e)




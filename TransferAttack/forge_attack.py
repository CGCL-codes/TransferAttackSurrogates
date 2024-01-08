'''
this file is for cifar10/100 only
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import argparse
import numpy as np
import torch
import torchattacks
import yaml
from tqdm import tqdm

from CIFAR_Train.utils import WrapModel



from CIFAR_Train import utils as util
import attacks


class TargetedIndexedDataset():
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes

    def __getitem__(self, idx):
        x, y, ii = self.dataset[idx]
        y += 1
        if y >= self.classes: y -= self.classes

        return x, y, ii

    def __len__(self):
        return len(self.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ResNet18',
                        help='choose the model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'cifar10-vit', 'tiny-imagenet', 'GTSRB'],
                        help='choose the dataset')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size')
    parser.add_argument('--cpu', action='store_true',
                        help='select to use cpu, otherwise use gpu')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='set the path to the exp data')
    parser.add_argument('--save-dir', type=str, default='./temp',
                        help='set which dictionary to save the experiment result')
    parser.add_argument('--save-name', type=str, default='temp-name',
                        help='set the save name of the experiment result')
    parser.add_argument('--resume', action='store_true',
                        help='set resume')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')
    parser.add_argument('--adv-type', type=str, default='pgd',
                        choices=['transfer', 'pgd', 'robust', 'fgsm-torchattack', 'fgsm-torchattack', 'pgd-torchattack','pgd-torchattack-l2', 'auto-torchattack', ])
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

    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'logit', 'doubleBP'])
    parser.add_argument('--targeted', action='store_true')

    parser.add_argument('--on-test-set', action='store_true')


    return parser.parse_args()


def regenerate_def_noise(def_noise, model, criterion, loader, attacker, cpu, logger):
    cnt = 0
    acc_total = 0
    rea_total = 0
    for x, y, ii in loader:
        cnt += 1
        logger.info('progress [{}/{}]'.format(cnt, len(loader)))
        if not cpu: x, y = x.cuda(), y.cuda()
        if 'torchattack' not in args.adv_type:
            def_x = attacker.perturb(model, criterion, x, y)
        else:
            def_x = attacker(x, y)
        with torch.no_grad():
            model.eval()
            _y = model(def_x)
            _y_ = model(x)
            rea_acc = (_y_.argmax(dim=1) == y).sum().item() / len(x)
            def_acc = (_y.argmax(dim=1) == y).sum().item() / len(x) if args.targeted else (_y.argmax(dim=1) != y).sum().item() / len(x)
            acc_total += def_acc
            rea_total += rea_acc
        logger.info('rea accuracy: {}'.format(rea_total / cnt))
        logger.info('fool accuracy: {}'.format(def_acc))
        logger.info('avg accuracy: {}'.format(acc_total / cnt))
        print(acc_total / cnt)
        def_noise[ii] = (def_x - x).detach().cpu().numpy() if 'l2' not in args.adv_type else def_x.detach().cpu().numpy()


def save_checkpoint(save_dir, save_name, model, log, def_noise=None):
    torch.save({
        'model_state_dict': util.get_model_state(model),
        }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f)
    if def_noise is not None:
        def_noise = (def_noise * 255).round()
        def_noise = def_noise.astype(np.int8)
        with open(os.path.join(save_dir, '{}-tap-noise.pkl'.format(save_name)), 'wb') as f:
            pickle.dump(def_noise, f)


def save_checkpoint_l2(save_dir, save_name, model, log, def_noise=None):
    torch.save({
        'model_state_dict': util.get_model_state(model),
    }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f)
    if def_noise is not None:
        def_noise = def_noise.astype(np.float64)
        with open(os.path.join(save_dir, '{}-tap-noise.pkl'.format(save_name)), 'wb') as f:
            pickle.dump(def_noise, f)


def main(args, logger):
    ''' init model / optim / loss func '''
    model = util.get_arch(args.arch, args.dataset, resume=True)
    normalize = util.get_normalize(args.dataset)

    if args.resume:
        state_dict = torch.load( os.path.join(args.resume_path) )
        model.load_state_dict( state_dict['model_state_dict'] )
        del state_dict

    wrap_model = WrapModel(model=model, normalizer=normalize)
    model = wrap_model
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    if args.on_test_set:
        train = False
    else:
        train = True

    train_loader = util.get_clean_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=train, fool=True, index=True)

    dataset = train_loader.loader.dataset
    ascending = True
    if args.targeted:
        print('targeted')
        if args.dataset == 'cifar10': classes = 10
        elif args.dataset == 'cifar100': classes = 100
        elif args.dataset == 'tiny-imagenet': classes = 200
        elif args.dataset == 'GTSRB': classes = 43
        else: raise ValueError
        dataset = TargetedIndexedDataset(dataset, classes)
        ascending = False
    train_loader = util.Loader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    if not args.cpu:
        wrap_model.cuda()
        criterion = criterion.cuda()

    if args.adv_type == 'pgd':
        attacker = attacks.PGDAttacker(
            radius=args.pgd_radius,
            steps=args.pgd_steps,
            step_size=args.pgd_step_size,
            random_start=args.pgd_random_start,
            norm_type=args.pgd_norm_type,
            ascending=ascending,
        )
    elif args.adv_type == 'transfer':
        attacker = attacks.TransferAttacker(
            radius=args.pgd_radius,
            steps=args.pgd_steps,
            step_size=args.pgd_step_size,
            random_start=False,
            norm_type=args.pgd_norm_type,
            ascending=not args.targeted,
            loss_type=args.loss_type,
            TI=True,
            DI=True,
            MI=True,
        )
    elif args.adv_type == 'robust':
        attacker = attacks.RobustMinimaxAttacker(
            ascending=ascending,
            radius=16,
            steps=200,
            step_size=2,
            reverse_radius=12,
            reverse_step_size=2,
            reverse_steps=6,
            awp=False
        )
    elif args.adv_type == 'fgsm-torchattack':
        attacker = torchattacks.FGSM(model=wrap_model, eps=args.pgd_radius / 255.)
    elif args.adv_type == 'pgd-torchattack':
        attacker = torchattacks.PGD(model=wrap_model, eps=args.pgd_radius / 255., steps=args.pgd_steps)
    elif args.adv_type == 'auto-torchattack':
        attacker = torchattacks.AutoAttack(model=wrap_model, eps=args.pgd_radius / 255.)
    elif args.adv_type == 'pgd-torchattack-l2':
        attacker = torchattacks.PGDL2(model=wrap_model, eps=args.pgd_radius, alpha=args.pgd_step_size, steps=args.steps)


    else: raise ValueError

    # if args.adv_type == 'pgd':
    #     attack = torchattacks.PGD
    #     attacker = attack(model=wrap_model, eps=args.pgd_radius / 255., alpha=args.pgd_step_size/255., steps=args.pgd_steps,
    #                       random_start=args.pgd_random_start)
    # elif args.adv_type == 'transfer':
    #     attack = torchattacks.TIFGSM
    #     attacker = attack(model=wrap_model, eps=args.pgd_radius / 255., alpha=args.pgd_step_size/255., steps=args.pgd_steps,
    #                       len_kernel=7, random_start=False)
    # else:
    #     raise NotImplementedError
    # if args.targeted:
    #     attacker.set_mode_targeted_random()

    ''' initialize the defensive noise (for unlearnable examples) '''
    data_nums = len(train_loader.loader.dataset)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'GTSRB':
        def_noise = np.zeros([data_nums, 3, 32, 32], dtype=np.float16)
    elif args.dataset == 'tiny-imagenet':
        def_noise = np.zeros([data_nums, 3, 64, 64], dtype=np.float16)
    elif args.dataset == 'imagenet-mini':
        def_noise = np.zeros([data_nums, 3, 256, 256], dtype=np.float16)
    else:
        raise NotImplementedError

    log = dict()
    logger.info('Noise generation started')
    regenerate_def_noise(
        def_noise, model, criterion, train_loader, attacker, args.cpu, logger)

    logger.info('Noise generation finished')

    save_checkpoint(args.save_dir, '{}'.format(args.save_name), model, log, def_noise) if 'l2' not in args.adv_type else save_checkpoint_l2(args.save_dir, '{}'.format(args.save_name), model, log, def_noise)
    test(args, logger)
    return


def evaluate_targeted(model, criterion, loader, cpu):
    acc = util.AverageMeter()
    loss = util.AverageMeter()
    model.eval()
    for x, y in tqdm(loader, total=len(loader)):
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == (y+1) % 10).sum().item() / len(x)
            lo = criterion(_y,(y+1) % 10).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


def test(args, logger):
    Test_Arch = ['resnet50', 'vgg16', 'inceptionv3', 'densenet121'] if args.dataset == 'cifar10' or args.dataset == 'GTSRB' else ['resnet50', 'vgg16_bn',  'densenet121']
    # Test_Arch = [ 'vgg16'] if args.dataset == 'cifar10' or args.dataset == 'GTSRB' else ['resnet50', 'vgg16_bn',  'densenet121']
    # Test_Arch = ['resnet50'] if args.dataset == 'cifar10' or args.dataset == 'GTSRB' else ['resnet50', 'vgg16_bn',  'densenet121']
    criterion = torch.nn.CrossEntropyLoss().cuda()
    # with open(os.path.join(args.save_dir, args.save_name+'.txt'), 'w') as f:
    #     f.write(args.save_dir.split('/')[-1] + '\n')
    for Arch in Test_Arch:
        logger.info("_________________")
        logger.info(Arch+":")
        print(Arch)
        # f.write(Arch + '\n')
        model = util.get_arch(Arch, args.dataset,resume=True)
        # resume_path = './CIFAR_Train/'+ args.dataset.lower() + '-models/{}-mixup-seed-40/model-fin-model.pkl'.format(Arch.lower().split('_')[0])
        # resume_path = './CIFAR_Train/'+ args.dataset.lower() + '-models/{}/model-fin-model.pkl'.format(Arch.lower().split('_')[0])
        resume_path = './CIFAR_Train/'+ args.dataset.lower() + '-models/{}-seed-40/model-fin-model.pkl'.format(Arch.lower().split('_')[0])
        state_dict = torch.load(resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model_state_dict'])
        del state_dict

        normalize = util.get_normalize(args.dataset)
        wrap_model = WrapModel(model=model, normalizer=normalize)
        model = wrap_model

        if not args.cpu:
            model.cuda()
        criterion = criterion.cuda()
        if args.on_test_set:
            train = False
        else:
            train = True
        train_loader = util.get_poisoned_loader(dataset=args.dataset, batch_size=args.batch_size, root=args.data_dir,
                                             train=train, noise_path=os.path.join(os.path.join(args.save_dir, '{}-tap-noise.pkl'.format(args.save_name))),fool=True)
        if args.targeted:
            test_acc, test_loss = evaluate_targeted(model, criterion, train_loader, args.cpu)
        else:
            test_acc, test_loss = util.evaluate(model, criterion, train_loader, args.cpu)
            test_acc = 1 - test_acc
        print("acc:{}, loss: {}".format(test_acc, test_loss))
        print("_________________")
        logger.info("acc:{}, loss: {}".format(test_acc, test_loss))
        logger.info('_________________')
        # logger.info("acc:{}, loss: {}\n".format(test_acc, test_loss))


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


if __name__ == '__main__':
    args = get_args()
    logger = util.generic_init(args)

    main(args, logger)
    # test(args, logger)
    # except Exception as e:
    #     logger.exception('Unexpected exception! %s', e)

"""
This code is partially based on the repository of https://github.com/locuslab/fast_adversarial (Wong et al., ICLR'20)
"""
import argparse
import logging
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
import torch.nn as nn
from robustness.robustbenchmaster.preact_resnet import PreActResNet18
from wideresnet import WideResNet
from utils import (upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd, evaluate_fgsm, l2_square, evaluate_standard)


def energy_xy(logits, y):
    return -torch.gather(logits, dim=1, index=y.view(y.shape[0], -1)).squeeze()

def energy_x(logits):
    return -torch.logsumexp(logits, dim=1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--out-dir', default='train_aaer_output', type=str)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0.0, type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'])
    parser.add_argument('--clamp', default=0, type=int)
    parser.add_argument('--lamda1', default=1.0, type=float)
    parser.add_argument('--lamda2', default=4.0, type=float)
    parser.add_argument('--lamda3', default=1.5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers = [
            logging.FileHandler(os.path.join(args.out_dir, 'output.log')),
            logging.StreamHandler()]
    )
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = ((args.epsilon * args.alpha) / 255.) / std

    lambda1 = args.lamda1
    lambda2 = args.lamda2
    lambda3 = args.lamda3

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=10).cuda()
    elif args.model == 'WideResNet':
        model = WideResNet(34, 10, 10, dropRate=0.0).cuda()
    else:
        raise ValueError("Unknown model")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[(lr_steps * 0.5), (lr_steps * 0.75)], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds  \t LR \t \t Train Loss \t Train Acc \t Ab NUM \t  Ab CE Loss \t Ab L2 Loss')
    for epoch in range(args.epochs):
        model.train()

        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        ae_num = 0
        ae_ce_loss = 0
        ae_l2_loss = 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if args.delta_init == 'zero':
                delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
            elif args.delta_init == 'random':
                delta_ran = torch.zeros(args.batch_size, 3, 32, 32).cuda()
                if args.clamp:
                    for j in range(len(epsilon)):
                        delta_ran[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                else:
                    for j in range(len(epsilon)):
                        delta_ran[:, j, :, :].uniform_(2 * -epsilon[j][0][0].item(), 2 * epsilon[j][0][0].item())
                delta = delta_ran

            delta.data = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            output = model(X + delta)
            logits = output.detach()
            loss = nn.CrossEntropyLoss(reduce=False)(output, y)
            loss_before = loss.detach()
            loss = loss.mean()
            loss.backward()
            grad = delta.grad.detach()
            delta.data = delta + alpha * torch.sign(grad)
            if args.clamp:
                delta.data = clamp(delta, -epsilon, epsilon)
            delta.data = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            delta.requires_grad = True
            output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
            adv_logits = output.detach()
            loss = nn.CrossEntropyLoss(reduce=False)(output, y)
            loss_after = loss.detach()
            loss = loss.mean()

            #abnormal samples are based on energy
            e_x = energy_x(logits).detach()
            e_xy= energy_xy(logits,y).detach().cpu()
            e_x_adv = energy_x(adv_logits).detach()
            e_xy_adv= energy_xy(adv_logits,y).detach().cpu()

            abnormal_example= e_x > e_x_adv 
            normal_example= e_x <= e_x_adv

            # abnormal_example = loss_before > loss_after
            # normal_example = loss_before <= loss_after
            abnormal_count = torch.count_nonzero(abnormal_example)
            normal_count = torch.count_nonzero(normal_example)
            ae_num += abnormal_count

            if abnormal_count != 0:
                abnormal_variation = l2_square(logits[abnormal_example], adv_logits[abnormal_example])
                abnormal_ce = abnormal_example * (e_x - e_x_adv)
                # abnormal_ce = abnormal_example * (loss_before - loss_after)
                abnormal_ce = abnormal_ce.sum() / abnormal_count

            if normal_count != 0:
                normal_variation = l2_square(logits[normal_example], adv_logits[normal_example])

            if epoch>0:
                if abnormal_count != 0 and normal_count != 0:
                    # loss = loss + (1 * abnormal_ce + 1 * max(abnormal_variation - normal_variation.item(), 0))
                    loss = loss +  (lambda1 * abnormal_count / y.size(0)) * (lambda2 * abnormal_ce + lambda3 * max(abnormal_variation - normal_variation.item(), 0)) # '* min((epoch/20), 1)' warm-up for long training schedule
                else:
                    loss = loss.mean()
            else:
                pass
            
            loss_energy = torch.mean(e_x - e_x_adv)
            loss_energy_xy = torch.mean(e_xy - e_xy_adv)
            delta = delta.detach()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %d \t \t %.4f \t %.4f', epoch, (epoch_time - start_epoch_time),
                    lr, (train_loss / train_n), (train_acc / train_n), ae_num, (ae_ce_loss / (ae_num + 0.0000000001)), (ae_l2_loss / (ae_num+ 0.0000000001)))
    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    # Evaluation
    if args.model == 'PreActResNet18':
        model_test = PreActResNet18(num_classes=10).cuda()
    elif args.model == 'WideResNet':
        model_test = WideResNet(34, 10, 10, dropRate=0.0).cuda()
    else:
        raise ValueError("Unknown model")

    model_test.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pth')))
    model_test.float()
    model_test.eval()

    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test, epsilon)
    pgd20_loss, pgd20_acc = evaluate_pgd(test_loader, model_test, 20, 10, epsilon)
    pgd50_loss, pgd50_acc = evaluate_pgd(test_loader, model_test, 50, 10, epsilon)

    logger.info('Test Loss \t Test Acc \t FGSM Loss \t FGSM Acc \t PGD20 Loss \t PGD20 Acc \t PGD50 Loss \t PGD50 Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, fgsm_loss, fgsm_acc, pgd20_loss, pgd20_acc, pgd50_loss, pgd50_acc)

    import json 
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump({
            'test_acc': test_acc, 
            'fgsm_acc': fgsm_acc, 
            'pgd20_acc': pgd20_acc, 
            'pgd50_acc': pgd50_acc}, f
        )


if __name__ == "__main__":
    main()

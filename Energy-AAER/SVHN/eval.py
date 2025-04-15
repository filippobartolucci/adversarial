"""
This code is partially based on the repository of https://github.com/locuslab/fast_adversarial (Wong et al., ICLR'20)
"""
import argparse
import logging
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from robustness.robustbenchmaster.preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd, evaluate_fgsm, l2_square, evaluate_standard)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def energy_xy(logits, y):
    return -torch.gather(logits, dim=1, index=y.view(y.shape[0], -1)).squeeze()

def energy_x(logits):
    return -torch.logsumexp(logits, dim=1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--model-dir', type=str) 
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader = get_loaders(args.data_dir, 32)

    epsilon = (args.epsilon / 255.) / std

    # Evaluation
    model_test = PreActResNet18(num_classes=100).cuda()
    model_test.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pth')))
    model_test.float()
    model_test.eval()
    model_test.to('cuda')

    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test, epsilon)
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 10, epsilon)

    print('Standard Test Loss: %.5f, Standard Test Accuracy: %.2f' % (test_loss, test_acc))
    print('FGSM Test Loss: %.5f, FGSM Test Accuracy: %.2f' % (fgsm_loss, fgsm_acc))
    print('PGD Test Loss: %.5f, PGD Test Accuracy: %.2f' % (pgd_loss, pgd_acc))

    import json
    with open(os.path.join(args.model_dir, 'results.json'), 'w') as f:
        json.dump({'real_acc': test_acc, 
                   'fgsm_acc': fgsm_acc, 
                   'pgd_acc': pgd_acc}, f
                   )

if __name__ == "__main__":
    main()

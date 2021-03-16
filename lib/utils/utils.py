import random
import torch
import numpy as np
import os
import copy
import torch.backends.cudnn as cudnn
from lib.configs.parse_arg import opt, args


def random_init(seed=0):
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.1, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def visualize_targets(preds, targets, i_batch, epoch=1):
    import matplotlib.pyplot as plt

    save_dir = '/root/medical/weak_semi_medical_seg/checkpoints/full0603/visualize/tmp_pced2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig = plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.set_title('preds')
    tmp = copy.deepcopy(preds[0])
    tmp[0, 0] = 0.99
    plt.imshow(tmp)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('1-preds')
    tmp = 1 - copy.deepcopy(preds[0])
    tmp[0, 0] = 0.99
    plt.imshow(tmp)

    ax = plt.subplot(2, 2, 3)
    ax.set_title('bg_mask')
    tmp = copy.deepcopy(targets[0, 0])
    tmp[0, 0] = 0.99
    plt.imshow(tmp)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('fg_mask')
    tmp = copy.deepcopy(targets[0, 1])
    tmp[0, 0] = 0.99
    plt.imshow(tmp)

    plt.savefig(save_dir + '/{}_{}.png'.format(epoch, i_batch))
    plt.close('all')

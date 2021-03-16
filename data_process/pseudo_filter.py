import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob

import ipdb

#####################################  UTILS  #####################################
def dice_func(pred, label, smooth=1e-8):
    inter_size = np.sum(((pred * label) > 0))
    sum_size = np.sum(pred) + np.sum(label)
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice

def get_metric(sample, reference):
    all_fg = (reference == 1).sum()
    tp = ((sample * reference) == 1).sum()
    fp = (np.logical_and(sample == 1, reference == 0)).sum()
    fn = (np.logical_and(sample == 0, reference == 1)).sum()

    tp_ratio, fp_ratio, fn_ratio, dice = tp*1.0/all_fg, fp*1.0/all_fg, fn*1.0/all_fg, dice_func(sample, reference)
    tp_ratio, fp_ratio, fn_ratio, dice = round(tp_ratio, 4), round(fp_ratio, 4), round(fn_ratio, 4), round(dice, 4)
    return {
        'TP': tp_ratio, 'FP': fp_ratio, 'FN': fn_ratio,
        'Dice': dice
    }

def get_seg_group(sample, reference, group='FP'):
    # return filtered indexs of a segmentation group, such as {FP, FN}
    if group == 'FP':
        indexs = np.logical_and(sample == 1, reference == 0)
    elif group == 'FN':
        indexs = np.logical_and(sample == 0, reference == 1)

    return indexs








#####################################  Methods  #####################################

def gen_prob_hist():
    # 0. src path, save path
    src_dir = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_train/'
    save_dir = src_dir
    gt_dir = src_dir + 'gt_label/'
    prob_dir = src_dir + 'heatmap/'
    gt_paths = sorted(glob(gt_dir + '*.nii.gz'))
    prob_paths = sorted(glob(prob_dir + '*.nii.gz'))
    length = len(gt_paths)

    # 1. load prob, gt
    terms = ['FP', 'FN']
    data = {term: np.array([]) for term in terms}
    for ith in range(length):
        gt_path, prob_path = gt_paths[ith], prob_paths[ith]
        gt, prob = nib.load(gt_path).get_data(), nib.load(prob_path).get_data()
        pred = (prob>0.5).astype(np.uint8)
        metrics = get_metric(pred, gt); print(metrics)
        for term in terms:
            values = prob[get_seg_group(pred, gt, term)]
            data[term] = np.append(data[term], values)

    for term in terms:
        plt.hist(data[term], bins=20); plt.title(term); # plt.show()    
        save_path = save_dir + 'prob_%s.png' % term
        plt.savefig(save_path); plt.cla()
            

    # 2. for each class {TP, FP, FN, TN}, plt save the histogram


if __name__ == "__main__":
    gen_prob_hist()
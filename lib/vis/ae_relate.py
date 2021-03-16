import numpy as np
import torch
import os
from glob import glob
import argparse
import nibabel as nib
from scipy import ndimage

import ipdb

def dice_func(pred, label, smooth=1e-8):
    inter_size = np.sum(((pred * label) == 1))
    sum_size = np.sum(pred==1) + np.sum(label==1)
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice
class Analyzer(object):
    '''
    function:
        1. 'get_metric':  obtain metrics of predictions,
        2. 'filter_pseudo':  filter some FG and BG regions for pseudo, return metrics
        3.

    '''
    def __init__(self, function='get_metric', terms=['pred'], params=None):
        self.metrics = {}
        self.params = params

        self.metrics_calculus = {}
        metrics_num = len(terms)
        for ith in range(metrics_num):
            self.metrics_calculus[ith] = {'TP': 0.0, 'FP': 0.0, 'FN': 0.0, 'Dice': 0.0}
        self.count = 0

    def prob2pseudo(self, prob, gt):
        # filter prob, save metrics, save filtered results
        fpr = self.params['fpr']; ftn = self.params['ftn']; fnr=self.params['fnr']


        pred = (prob>0.5).astype(np.uint8)
        metrics = self.get_metric(pred, gt)
        self.metrics[0] = metrics

        # filtered
        pseudo = pred.copy()
        pos_values = sorted(prob[prob>0.5])
        out_num = int(len(pos_values)*fp_filter_ratio)
        pos_value = pos_values[out_num]
        neg_values = sorted(prob[prob<=0.5], reverse=True)
        neg_value = neg_values[int(out_num*fn_filter_ratio)]
        pseudo[np.logical_and(pseudo==1, prob<pos_value)] = 255
        pseudo[np.logical_and(pseudo==0, prob>neg_value)] = 255
        metrics = self.get_metric(pseudo, gt)
        self.metrics[1] = metrics

        self.accumulate(self.metrics)
        return pseudo, self.metrics

    def analyze(self, samples, reference):
        '''
        :param samples:     multiple inputs for calculating metrics
        :param reference:   gt mask
        :return:            metrics
        '''
        output = None
        # metric: separate metrics
        for ith, term in enumerate(self.terms):
            if ith < len(samples):
                sample = samples[ith]
            if term == 'intersect':
                sample = np.logical_and(samples[0] == 1, samples[1] == 1)
            elif term == 'union':
                sample = np.logical_or(samples[0] == 1, samples[1] == 1)
            metric = self.get_metric(sample, reference)
            self.metrics[ith] = metric
        self.accumulate(self.metrics)
        return output, self.metrics

    def reset(self):
        for ith in range(self.num_samples+2):
            self.metrics_calculus[ith] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.count = 0

    def get_metric(self, sample, reference):
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

    def accumulate(self, metrics):
        self.count += 1
        for key, value in metrics.items():
            for kk, vv in value.items():
                self.metrics_calculus[key][kk] += vv

    def get_avg(self):
        metrics = self.metrics.copy()
        for key, value in self.metrics_calculus.items():
            for kk, vv in value.items():
                metrics[key][kk] = round(self.metrics_calculus[key][kk] / self.count, 4)

        return metrics

def calculate_avg_total_num(_dir):
    '''
    Average volume total num for samples
    '''
    paths = sorted(glob(_dir+'*.nii.gz'))
    total_nums = np.array([0.0 for _ in paths])
    for ith, path in enumerate(paths):
        sample = nib.load(path).get_data()
        total_nums[ith] = sample.sum()
    return total_nums.mean(), total_nums

def stats_num():
    dir_ae = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0825_trachea_aelo_11_train/pred/'
    avg_num, total_nums = calculate_avg_total_num(dir_ae)

    print('Average total num: ', avg_num)
    print('Total num: ', sorted(total_nums))

if __name__ == '__main__':
    stats_num()


    '''
    from .analysis import Analyzer
    Params = {
        'k': 1.0, 'fpr': 0.3,
        'total_num': 7813.0,    # averaged total fg num
    }

    # 1. load probabilities
    dir_ae_prob = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0825_trachea_aelo_11_train/heatmap/'
    dir_ae_gt = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0825_trachea_aelo_11_train/gt_label/'
    prob_paths = sorted(glob(dir_ae_prob + '*.nii.gz'))
    gt_paths = sorted(glob(dir_ae_gt + '*.nii.gz'))

    # 2. filter params and filter save
    for ith, (prob_path, gt_path) in enumerate(zip(prob_paths, gt_paths)):
        prob, gt = nib.load(prob_path).get_data(), nib.load(gt_path).get_data()
        filter_total_num = Params['k'] * Params['total_num']
        # pseudo, metrics = filter2pseudo(prob, gt, fpr=Params['fpr'], ftn=filter_total_num)
        # save pseudo, log write

    # log avg, log close

    # 3. comparing
    '''
import numpy as np
import torch
import os
from glob import glob
import argparse
import nibabel as nib
from scipy import ndimage

import ipdb

import matplotlib.pyplot as plt
def plot_hist_cdf(values, type='pos', gt=None, pos_nums=None, title_prefix='',
                  bins=50, pos_range=(0.5, 1.0), neg_range=(0.0, 0.5),
                  ):
    '''
    :param values:  list of probabilities
    :param type:    {'pos', 'neg'}
    :return:        plot or save figure containing histogram and cdf
    '''
    if type == 'pos':
        pos_values = values
        pos_values = sorted(pos_values)
        width = (pos_range[1] - pos_range[0]) / bins

        # 1. plot hist
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('pixel_nums')
        counts, bins, bars = ax1.hist(pos_values, bins=bins, range=pos_range)
        # 1.1 disable last bar
        plt.cla()
        counts_ = counts.copy(); counts_[-1] = 0
        ax1.bar(bins[:-1], counts_, width=width)

        # 2. cumulative ratio
        ax2 = ax1.twinx()
        ax2.set_ylabel('cdf')
        counts_ratios = counts / pos_nums
        counts_cdf = np.array([counts_ratios[:ith].sum() for ith in range(1, len(counts_ratios) + 1)])
        ax2.plot(bins[:-1], counts_cdf, color='g')
        # 2.1 plot texts
        for ith, (a, b) in enumerate(zip(bins[:-1], counts_cdf)):
            if ith % 5 == 0 or ith == (len(bins)-3):
                plt.text(a, b, str('%.2f' % b))

        # 3. show
        plt.title('%sPostive histogram and cdf' % title_prefix)
        if Params['save']:
            plt.savefig(Params['save_dir'] + '%spos_hist.png' % title_prefix)
        else:
            plt.show()
        plt.close('all')

        # 4. TP & FP
        TP_ratios, FP_ratios = np.zeros_like(counts_ratios), np.zeros_like(counts_ratios)
        for ith, start in enumerate(bins[:-1]):
            if ith == len(bins[:-1])-1:
                gt_bin = gt[np.logical_and(values >= bins[ith], values <= bins[ith + 1])]
            else:
                gt_bin = gt[np.logical_and(values >= bins[ith], values < bins[ith+1])]
            TP_ratios[ith] = (gt_bin == 1).astype(np.uint8).sum() * 1.0 / pos_nums
            FP_ratios[ith] = (gt_bin == 0).astype(np.uint8).sum() * 1.0 / pos_nums
        TP_cdf = np.array([TP_ratios[:ith].sum() for ith in range(1, len(TP_ratios) + 1)])
        FP_cdf = np.array([FP_ratios[:ith].sum() for ith in range(1, len(FP_ratios) + 1)])
        # 4.1 plot bars
        fig2, ax3 = plt.subplots()
        ax3.bar(bins[:-1], TP_cdf, width=width/2, label='TP')
        ax3.bar(bins[:-1]+width/2, FP_cdf, width=width/2, label='FP')
        # 4.2 plot texts
        for ith, (a, b) in enumerate(zip(bins[:-1], FP_cdf)):
            if ith % 5 == 0 or ith == (len(bins)-3):
                plt.text(a, b, str('%.2f' % b))
        # 5. show
        ax3.legend()
        plt.title('%sPostive histogram: TP vs. FP' % title_prefix)
        if Params['save']:
            plt.savefig(Params['save_dir'] + '%spos_TPFP.png' % title_prefix)
        else:
            plt.show()
        plt.close('all')

    elif type == 'neg':
        neg_values = values
        neg_values = sorted(neg_values, reverse=True)
        width = (neg_range[1] - neg_range[0]) / bins

        # 1. plot hist
        fig, ax1 = plt.subplots()
        ax1.set_ylabel('pixel_nums')
        counts, bins, bars = ax1.hist(neg_values, bins=bins, range=neg_range)
        # 1.1 disable last bar
        plt.cla()
        counts = counts[::-1]; bins = bins[::-1]
        counts_ = counts.copy(); counts_[-1] = 0
        ax1.bar(bins[:-1], counts_, width=width)
        ax1.invert_xaxis()

        # 2. cumulative ratio
        ax2 = ax1.twinx()
        ax2.set_ylabel('cdf')
        counts_ratios = counts / (pos_nums * 2)
        counts_cdf = np.array([counts_ratios[:ith].sum() for ith in range(1, len(counts_ratios) + 1)])
        counts_cdf[counts_cdf > 1.0] = 1.0
        ax2.plot(bins[:-1], counts_cdf, color='g')
        for ith, (a, b) in enumerate(zip(bins[:-1], counts_cdf)):
            if ith % 5 == 0 or ith == (len(bins)-3):
                plt.text(a, b, str('%.2f' % b))
        plt.title('%sNegative histogram and cdf' % title_prefix)
        if Params['save']:
            plt.savefig(Params['save_dir'] + '%sneg_hist' % title_prefix)
        else:
            plt.show()
        plt.close('all')

        # 3. FN
        FN_ratios = np.zeros_like(counts_ratios)
        for ith, start in enumerate(bins[:-1]):
            gt_bin = gt[np.logical_and(values < bins[ith], values >= bins[ith + 1])]
            FN_ratios[ith] = (gt_bin == 1).astype(np.uint8).sum() * 1.0 / (pos_nums * 2)
        FN_cdf = np.array([FN_ratios[:ith].sum() for ith in range(1, len(FN_ratios) + 1)])

        # 4.1 plot bars
        fig2, ax3 = plt.subplots()
        ax3.bar(bins[:-1], FN_cdf, width=width / 2, label='FN')
        ax3.invert_xaxis()
        # 4.2 plot texts
        for ith, (a, b) in enumerate(zip(bins[:-1], FN_cdf)):
            if ith % 5 == 0 or ith == (len(bins) - 3):
                plt.text(a, b, str('%.2f' % b))
        # 5. show
        ax3.legend()
        plt.title('%sNegative histogram: FN' % title_prefix)
        if Params['save']:
            plt.savefig(Params['save_dir'] + '%sneg_FN.png' % title_prefix)
        else:
            plt.show()
        plt.close('all')

def show_ranking_metric():
    for sample in samples:
        if Params['source'] == 'prob':
            postfix = '_00.nii.gz' if os.path.exists(dir_prob + '%s_00.nii.gz' % sample) else '.nii.gz'
            path_prob = dir_prob + sample + postfix
            prob = nib.load(path_prob).get_data()
            bins = 50
            pos_range = (0.99, 1.0)
            neg_range = (0.0, 0.0001)
            # pos_range = (0.5, 1.0)
            # neg_range = (0.0, 0.5)
            if Params['method'] in ['pweight', 'prob_rank']:
                pos_range = (0.0, 1.0) # (0.5, 1.0)
                neg_range = (0.0, 1.0) # (0.0, 0.5)
        elif Params['source'] == 'ae':
            postfix = '_00.nii.gz' if os.path.exists(dir_ae + '%s_00.nii.gz' % sample) else '.nii.gz'
            path_ae = dir_ae + sample + postfix
            prob = nib.load(path_ae).get_data()
            bins = 50
            pos_range = (0.99, 1.0)
            neg_range = (0.0, 0.0001)

        path_gt = dir_gt + '%s%s' % (sample, postfix)
        gt = nib.load(path_gt).get_data()
        pred = (prob > 0.5).astype(np.uint8)

        # split pos/neg pixels, ranking
        if Params['method'] in ['vanilla', 'prob_rank']:
            pos_values = prob[prob > 0.5].flatten()
            neg_values = prob[prob <= 0.5].flatten()
            pos_gt = gt[prob > 0.5].flatten()
            neg_gt = gt[prob <= 0.5].flatten()
            pos_nums = pred.sum() * 1.0

            if Params['method'] in ['prob_rank']:
                # pos_values, pos_indices = torch.sort(pos_values)
                # neg_values, neg_indices = torch.sort(neg_values, descending=True)
                # pos_rank_prob = torch.zeros_like(pos_values, dtype=torch.long)
                # neg_rank_prob = torch.zeros_like(neg_values, dtype=torch.long)
                # pos_rank_prob[pos_indices] = torch.arange(len(pos_indices), dtype=torch.long)
                # neg_rank_prob[neg_indices] = torch.arange(len(neg_indices), dtype=torch.long)
                pos_indices = np.argsort(pos_values)
                neg_indices = np.argsort(neg_values)[::-1]
                pos_rank_prob = np.zeros_like(pos_values, dtype=np.long)
                neg_rank_prob = np.zeros_like(neg_values, dtype=np.long)
                pos_rank_prob[pos_indices] = np.arange(len(pos_indices), dtype=np.long)
                neg_rank_prob[neg_indices] = np.arange(len(neg_indices), dtype=np.long)

                # positive: ranking rescale [0,1]
                Max_rank = len(pos_rank_prob)
                pos_values = pos_rank_prob * 1.0 / Max_rank

                # negative: uppper bound 2*Max_rank
                neg_rank_prob[neg_rank_prob > 2*Max_rank] = 2*Max_rank
                neg_values = neg_rank_prob * 1.0 / (2*Max_rank)

        elif Params['method'] in ['pweight']:
            path_mask = dir_mask + sample + postfix
            mask = nib.load(path_mask).get_data()
            pos_values = prob[mask == 1].flatten()
            neg_values = prob[mask == 0].flatten()
            pos_gt = gt[mask == 1].flatten()
            neg_gt = gt[mask == 0].flatten()
            pos_nums = mask.sum() * 1.0

        prefix = '%s_%s_' % (Params['source'], sample)
        if pos_range[0] > 0.5:
            pos_values[pos_values < pos_range[0]] = pos_range[0]
        if neg_range[0] > 0.5:
            neg_values[neg_values > neg_range[1]] = neg_range[1]
        plot_hist_cdf(pos_values, type='pos', gt=pos_gt, pos_nums=pos_nums, title_prefix=prefix,
                      bins=bins, pos_range=pos_range, neg_range=neg_range)
        plot_hist_cdf(neg_values, type='neg', gt=neg_gt, pos_nums=pos_nums, title_prefix=prefix,
                      bins=bins, pos_range=pos_range, neg_range=neg_range)

if __name__ == '__main__':

    Params = {
        'method': 'vanilla',   # {'prob_rank', 'pweight', 'vanilla'}

        'source': 'prob',      # {'prob', 'ae'}
        'save':    True,       # {True, False}
        'group':  'tra_r1',    # {'pro_r1', 'pro_r3', 'tra_r1', 'tra_r3'}
    }
    Params['save_dir'] = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/tmp/prob_ranking/%s/' % Params['group']
    if not os.path.exists(Params['save_dir']):
        os.makedirs(Params['save_dir'])

    if Params['method'] in ['vanilla', 'prob_rank']:
        dir_prob = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/1013_tra_aelo_13_train/heatmap/'
        dir_mask = ''
    elif Params['method'] in ['pweight']:
        dir_prob = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/1016_debug_train/pweight/'
        dir_mask = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/1016_debug_train/pred/'

    dir_ae = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/1013_pro_aelo_33_train/heatmap/'
    dir_gt = dir_prob[:-8] + '/gt_label/'

    lib_samples = {
        'pro_r1': ['Case00', 'Case09', 'Case17', 'Case23'],
        'pro_r3': ['Case00', 'Case08', 'Case03', 'Case23'],
        'tra_r1': ['Case01', 'Case08', 'Case14', 'Case30'],
        'tra_r3': ['Case00', 'Case08', 'Case03', 'Case23'],
    }
    samples = lib_samples[Params['group']]
    show_ranking_metric()
    # ipdb.set_trace()




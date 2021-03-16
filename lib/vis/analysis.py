import numpy as np
import torch
import os
from glob import glob
import argparse
import nibabel as nib
from scipy import ndimage
from scipy.io import loadmat
import math
from datetime import datetime
import ipdb


def dice_func(pred, label, smooth=1e-8):
    inter_size = np.sum(((pred * label) == 1))
    sum_size = np.sum(pred==1) + np.sum(label==1)
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice

class Gaussian2D():
    def __init__(self, mean=(0, 0), variance=(1, 1)):
        self.mean = mean
        self.variance = variance

    def forward(self, inputs1, inputs2):
        mu1, mu2 = self.mean
        sigma1, sigma2 = self.variance
        part1 = 1.0 / (2*math.pi*sigma1*sigma2)
        part2 = torch.pow((inputs1 - mu1), 2)/(sigma1**2) + torch.pow((inputs2 - mu2), 2)/(sigma2**2)
        results = part1 * torch.exp(-1.0/2 * part2)
        return results

from scipy.stats import multivariate_normal
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
def compute_sdf(arr, truncate_value=20):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(t) = 0; t in segmentation boundary
             +inf|t-b|; x in segmentation
             -inf|t-b|; x out of segmentation
    normalize sdf to [-1,1]
    """

    posmask = arr.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        posdis[posdis > truncate_value] = truncate_value
        negdis[negdis > truncate_value] = truncate_value
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        tsdf = (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis)) - \
              (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis))
        tsdf[boundary == 1] = 0

    return tsdf #, np.max(posdis), np.max(negdis)

class Analyzer(object):
    '''
    1. INIT
        default num_samples = 2
    2. Analyze
        samples, reference: binary labels

    '''
    def __init__(self, num_samples=2, terms=None, weak_refine=False, gt_replace_boundary=False, prob_filter=False, params=None):
        self.num_samples = num_samples
        self.terms = terms      # ['main_pred', 'cmp1', 'intersect']
        self.metrics = {}
        self.weak_refine = weak_refine
        self.gt_replace_boundary = gt_replace_boundary
        self.prob_filter = prob_filter
        self.params = params

        # x1, x2, intersection, union
        self.metrics_calculus = {}
        metrics_num = self.num_samples if prob_filter else len(terms)
        for ith in range(metrics_num):
            self.metrics_calculus[ith] = {'TP': 0.0, 'FP': 0.0, 'FN': 0.0, 'TN': 0.0, 'Dice': 0.0}
        self.count = 0

    def prob2pseudo(self, prob, gt):
        # filter prob, save metrics, save filtered results
        fp_filter_ratio = self.params['fp_filter_ratio']
        fn_filter_ratio = self.params['fn_filter_ratio']
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

    def uncertain2pseudo(self, sources, gt):
        # filter fused prob, save metrics, save filtered results
        fp_filter_ratio = self.params['fp_filter_ratio']
        fn_times = self.params['fn_filter_ratio']
        v_uncertain, pred, sdf = sources
        metrics = self.get_metric(pred, gt)
        self.metrics[0] = metrics

        # filtered
        pseudo = pred.copy()
        pos_values = sorted(v_uncertain[pred == 1], reverse=True);
        out_num = int(len(pos_values) * fp_filter_ratio)
        pos_value = pos_values[out_num]; print(pos_value)
        neg_values = sorted(v_uncertain[pred == 0], reverse=True);
        neg_value = neg_values[int(out_num*fn_times)]; print(neg_value)
        pseudo[np.logical_and(pseudo == 1, v_uncertain > pos_value)] = 255
        pseudo[np.logical_and(pseudo == 0, v_uncertain > neg_value)] = 255
        # employ sdf to filter FP: pred==1, sdf<0
        pseudo[np.logical_and(pseudo == 1, sdf<0)] = 0
        # pseudo[np.logical_and(pseudo==1, sdf<=-1)] = 0

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
        tn = (np.logical_and(sample == 0, reference == 0)).sum()

        tp_ratio, fp_ratio, fn_ratio, dice = tp*1.0/all_fg, fp*1.0/all_fg, fn*1.0/all_fg, dice_func(sample, reference)
        tp_ratio, fp_ratio, fn_ratio, dice = round(tp_ratio, 4), round(fp_ratio, 4), round(fn_ratio, 4), round(dice, 4)
        tn_ratio = tn*1.0/all_fg; tn_ratio = round(tn_ratio, 4)
        return {
            'TP': tp_ratio, 'FP': fp_ratio, 'FN': fn_ratio, 'TN': tn_ratio,
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

def getLargestCC(segmentation):
    labels, num_features = ndimage.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(np.uint8)

def calculate_metric():
    '''
    Param: exp_id       
    Param: phase        {'train', 'val'}
    Param: source       {'best', 'em_init'}
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default='', type=str, help="")
    ap.add_argument("--phase", default='train', type=str, help="phase: {'train', 'val'}")
    ap.add_argument("--source", default='best', type=str, help="source: {'best', 'em_init'}")
    ap.add_argument("--gt_id", default='', type=str, help="exp_id: where is gt label from")
    args = vars(ap.parse_args())

    # directories
    exp_id, phase, source = args['exp_id'], args['phase'], args['source']
    dir_prefix = '/group/lishl/weak_exp/output/%s_%s/' % (exp_id, phase)
    if args['gt_id'] == '':
        dir_gt = dir_prefix + 'gt_label/'
    else:
        dir_gt = '/group/lishl/weak_exp/output/%s_%s/' % (args['gt_id'], phase) + 'gt_label/'

    dir_pred = None
    if source == 'best':
        dir_pred = dir_prefix + 'pred/'
    elif source == 'em_init':
        dir_pred = '/group/lishl/weak_exp/em_save_pseudo/%s/vis/' % exp_id
    else:
        raise NotImplementedError

    # paths and log
    log_path = dir_pred + '/log_metric.txt'
    source_paths = {
        'gt': sorted(glob(dir_gt + '*.nii.gz')),
        'pred': sorted(glob(dir_pred + '*.nii.gz')),
    }
    log = open(log_path, 'w')

    # get metric
    analyzer = Analyzer(terms=['pred'])
    for ith in range(len(source_paths['gt'])):
        gt = nib.load(source_paths['gt'][ith]).get_data()
        pred = nib.load(source_paths['pred'][ith]).get_data()
        _, metrics = analyzer.analyze([pred], gt)
        # log write
        name = (source_paths['pred'][ith].split('/')[-1]).split('.')[0]
        if len(metrics.keys()) == 1:    # only one group metric
            log.write('%s\t' % name)
            for key, value in metrics.items():
                for subkey, subvalue in value.items():
                    log.write('%s: %.4f  ' % (subkey, subvalue))
                log.write('\n')
        else:                           # multiple group metrics
            log.write('%s\n' % name)
            for key, value in metrics.items():
                log.write('%s:\t %s\n' % (str(key), str(value)))
            log.write('\n')

    # log write avg
    log.write('\nAverage Metrics:\n')
    avg_metric = analyzer.get_avg()
    for key, value in avg_metric.items():
        log.write('%s:\t %s\n' % (str(key), str(value)))
    log.close()


def filter_to_pseudo():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default='prob', type=str, help="{'prob', 'pred'}")
    ap.add_argument("--dir", default='0723_trachea_balance_r3_fgw1_auto_ani_train', type=str, help="target directory")
    ap.add_argument("--fpr", default=0.5, type=float, help="fp_filter_ratio")
    ap.add_argument("--fnr", default=2.0, type=float, help="fn_filter_ratio")

    args = vars(ap.parse_args())
    params = {
        'fp_filter_ratio': args['fpr'], # 0.5,
        'fn_filter_ratio': args['fnr'], # 0.01
    }

    source_key, dir = args['source'], args['dir']
    dir_prefix = '/group/lishl/weak_exp/output/'
    gt_paths = sorted(glob(dir_prefix + dir + '/gt_label/*.nii.gz'))

    if source_key == 'prob':
        log_path = dir_prefix + dir + '/log_process_%s_%.1f_%.1f.txt' % (source_key, params['fp_filter_ratio'], params['fn_filter_ratio'])
        pseudo_dir = dir_prefix + dir + '/pseudo_%.1f_%.1f/' % (params['fp_filter_ratio'], params['fn_filter_ratio'])
        source_paths = sorted(glob(dir_prefix + dir + '/heatmap/*.nii.gz'))
    elif source_key == 'pred':
        log_path = dir_prefix + dir + '/log_process_%s_%d.txt' % (source_key, params['fg_erode_iter'])
        pseudo_dir = dir_prefix + dir + '/pseudo_erode%d/' % params['fg_erode_iter']
        source_paths = sorted(glob(dir_prefix + dir + '/pred/*.nii.gz'))
    log = open(log_path, 'w')
    if not os.path.exists(pseudo_dir):
        os.makedirs(pseudo_dir)

    analyzer = Analyzer(prob_filter=True, params=params)
    for ith in range(len(gt_paths)):
        src = nib.load(source_paths[ith]).get_data()
        gt = nib.load(gt_paths[ith]).get_data()
        if source_key == 'prob':
            pseudo, metrics = analyzer.prob2pseudo(src, gt)
        elif source_key == 'pred':
            pseudo, metrics = analyzer.erode2pseudo(src, gt)
        # save pseudo label
        filename = gt_paths[ith].split('/')[-1]
        nib.save(nib.Nifti1Image(pseudo, np.eye(4)), pseudo_dir+filename)
        # log write
        name = (gt_paths[ith].split('/')[-1]).split('_00')[0]
        log.write('%s\n' % name)
        for key, value in metrics.items():
            log.write('%s:\t %s\n' % (str(key), str(value)))
        log.write('\n')

        # log write avg
    log.write('\nAverage Metrics:\n')
    avg_metric = analyzer.get_avg()
    for key, value in avg_metric.items():
        log.write('%s:\t %s\n' % (str(key), str(value)))
    log.close()

def gen_uncertainty_volume():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--prob_dir", default='0723_trachea_balance_r3_fgw1_auto_ani_train', type=str, help="prob source directory")
    ap.add_argument("-ds", "--sdf_dir", default='0723_trachea_balance_r3_fgw1_auto_ani_train', type=str, help="tsdf source directory")
    args = vars(ap.parse_args())
    prob_dir, sdf_dir = args['prob_dir'], args['sdf_dir']
    dir_prefix = '/group/lishl/weak_exp/output/'
    pseudo_dir = dir_prefix + prob_dir + '/probsdf/'
    save_dir_sdf = dir_prefix + prob_dir + '/ae_sdf/'

    if not os.path.exists(pseudo_dir):
        os.makedirs(pseudo_dir)
    if not os.path.exists(save_dir_sdf):
        os.makedirs(save_dir_sdf)

    source_paths = {
        'prob': sorted(glob(dir_prefix + prob_dir + '/heatmap/*.nii.gz')),
        'sdf': sorted(glob(dir_prefix + sdf_dir + '/pred/*.nii.gz')),
    }
    mean, variance = (0.5, 0), (0.5 / 3, 1.0 / 3)
    gauss = Gaussian2D(mean, variance)

    for ith in range(len(source_paths['prob'])):
        src_prob = nib.load(source_paths['prob'][ith]).get_data()
        src_sdf = nib.load(source_paths['sdf'][ith]).get_data()
        src_sdf = compute_sdf(src_sdf)

        filename = source_paths['prob'][ith].split('/')[-1]
        nib.save(nib.Nifti1Image(src_sdf, np.eye(4)), save_dir_sdf + filename)

        prob, sdf = torch.from_numpy(src_prob), torch.from_numpy(src_sdf)
        v_uncertain = gauss.forward(prob, sdf)
        v_uncertain = v_uncertain.numpy()

        filename = source_paths['prob'][ith].split('/')[-1]
        nib.save(nib.Nifti1Image(v_uncertain, np.eye(4)), pseudo_dir + filename)


def gen_pseudo_with_prob_dist():
    ap = argparse.ArgumentParser()
    ap.add_argument("-onae", "--onae", default=0, type=int, help="pseudo label from ae pred or weak baseline pred")
    ap.add_argument("-dp", "--prob_dir", default='0723_trachea_balance_r3_fgw1_auto_ani_train', type=str, help="prob source directory")
    ap.add_argument("-da", "--ae_dir", default='', type=str, help="ae source directory")
    ap.add_argument("--fpr", default=0.5, type=float, help="fp_filter_ratio")
    ap.add_argument("--fnr", default=1.0, type=float, help="N times of fpr filtered pixels")

    args = vars(ap.parse_args())
    params = {
        'fp_filter_ratio': args['fpr'], # 0.5,
        'fn_filter_ratio': args['fnr'], # 0.01,
    }
    source_key = 'probsdf'

    # m1. load prob for baseline, compute tsdf for ae_pred
    # m2. combine two source, obtain new uncertainty on gaussian function
    # m3. filter those with high uncertainty
    # m4. save result, write metric log

    prob_dir = args['prob_dir']
    save_dir = args['ae_dir'] if args['onae'] else prob_dir
    dir_prefix = '/group/lishl/weak_exp/output/'
    log_path = dir_prefix + save_dir + '/log_process_%s_%.1f_%.1f.txt' % (source_key, params['fp_filter_ratio'], params['fn_filter_ratio'])
    pseudo_dir = dir_prefix + save_dir + '/pseudo_probsdf_%.1f_%.1f/' % (params['fp_filter_ratio'], params['fn_filter_ratio'])
    source_paths = {
        'gt': sorted(glob(dir_prefix + prob_dir + '/gt_label/*.nii.gz')),
        'pred': sorted(glob(dir_prefix + save_dir + '/pred/*.nii.gz')),
        'sdf': sorted(glob(dir_prefix + prob_dir + '/ae_sdf/*.nii.gz')),
        'probsdf': sorted(glob(dir_prefix + prob_dir + '/probsdf/*.nii.gz')),
    }

    log = open(log_path, 'w')
    if not os.path.exists(pseudo_dir):
        os.makedirs(pseudo_dir)

    analyzer = Analyzer(prob_filter=True, params=params)    # modify
    for ith in range(len(source_paths['gt'])):
        gt = nib.load(source_paths['gt'][ith]).get_data()
        src_pred = nib.load(source_paths['pred'][ith]).get_data()
        src_sdf = nib.load(source_paths['sdf'][ith]).get_data()
        src_probsdf = nib.load(source_paths['probsdf'][ith]).get_data()

        # prob + sdf -> uncertainty
        pseudo, metrics = analyzer.uncertain2pseudo([src_probsdf, src_pred, src_sdf], gt)

        # save pseudo label
        filename = source_paths['gt'][ith].split('/')[-1]
        nib.save(nib.Nifti1Image(pseudo, np.eye(4)), pseudo_dir+filename)
        # log write
        name = (source_paths['gt'][ith].split('/')[-1]).split('_00')[0]
        log.write('%s\n' % name)
        for key, value in metrics.items():
            log.write('%s:\t %s\n' % (str(key), str(value)))
        log.write('\n')

        # log write avg
    log.write('\nAverage Metrics:\n')
    avg_metric = analyzer.get_avg()
    for key, value in avg_metric.items():
        log.write('%s:\t %s\n' % (str(key), str(value)))
    log.close()

def show_fp_fn():

    def get_layout_fpfn(v_uncertain, pred, gt, phase='prob', sdf=None):
        layout = np.zeros_like(pred, dtype=np.uint8)
        label_fpfn = np.zeros_like(pred, dtype=np.uint8)
        pseudo = pred.copy()
        if phase == 'prob':
            pos_values = sorted(v_uncertain[pred == 1])
            out_num = int(len(pos_values) * fp_filter_ratio)
            pos_value = pos_values[out_num]
            neg_values = sorted(v_uncertain[pred == 0], reverse=True)
            neg_value = neg_values[int(out_num * fn_times)]
            layout[np.logical_and(pred == 1, v_uncertain < pos_value)] = labelmap['filter_pos']
            layout[np.logical_and(pred == 0, v_uncertain > neg_value)] = labelmap['filter_neg']
            pseudo[np.logical_and(pred == 1, v_uncertain < pos_value)] = 255
            pseudo[np.logical_and(pred == 0, v_uncertain > neg_value)] = 255
        elif phase == 'probsdf':
            pos_values = sorted(v_uncertain[pred == 1], reverse=True)
            out_num = int(len(pos_values) * fp_filter_ratio)
            pos_value = pos_values[out_num]
            neg_values = sorted(v_uncertain[pred == 0], reverse=True)
            neg_value = neg_values[int(out_num * fn_times)]
            layout[np.logical_and(pred == 1, v_uncertain > pos_value)] = labelmap['filter_pos']
            layout[np.logical_and(pred == 0, v_uncertain > neg_value)] = labelmap['filter_neg']
            pseudo[np.logical_and(pred == 1, v_uncertain > pos_value)] = 255
            pseudo[np.logical_and(pred == 0, v_uncertain > neg_value)] = 255
            # employ sdf to filter FP: pred==1, sdf<0
            pseudo[np.logical_and(pred == 1, sdf < 0)] = 0
        label_fpfn[np.logical_and(gt == 0, pseudo == 1)] = labelmap['fp']
        label_fpfn[np.logical_and(gt == 1, pseudo == 0)] = labelmap['fn']
        return layout, label_fpfn

    # argparser: read params
    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--prob_dir", default='0723_trachea_balance_r3_fgw1_auto_ani_train', type=str,
                    help="prob source directory")
    ap.add_argument("--fpr", default=0.5, type=float, help="fp_filter_ratio")
    ap.add_argument("--fnr", default=2.0, type=float, help="N times of fpr filtered pixels")
    args = vars(ap.parse_args())
    fp_filter_ratio, fn_times = args['fpr'], args['fnr']

    prob_dir = args['prob_dir']
    dir_prefix = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/'
    source_paths = {
        'gt': sorted(glob(dir_prefix + prob_dir + '/gt_label/*.nii.gz')),
        'pred': sorted(glob(dir_prefix + prob_dir + '/pred/*.nii.gz')),
        'sdf': sorted(glob(dir_prefix + prob_dir + '/ae_sdf/*.nii.gz')),
        'prob': sorted(glob(dir_prefix + prob_dir + '/heatmap/*.nii.gz')),
        'probsdf': sorted(glob(dir_prefix + prob_dir + '/probsdf/*.nii.gz')),
    }
    cmp_uncertain_dir = dir_prefix + prob_dir + '/pseudo_compare_fpfn_%.1f/' % args['fpr']
    if not os.path.exists(cmp_uncertain_dir):
        os.makedirs(cmp_uncertain_dir)

    # load gt, prob or probsdf
    labelmap = {'filter_pos': 2, 'filter_neg': 3, 'fp': 1, 'fn': 2}
    for ith in range(len(source_paths['gt'])):
        gt = nib.load(source_paths['gt'][ith]).get_data()
        src_pred = nib.load(source_paths['pred'][ith]).get_data()
        src_prob = nib.load(source_paths['prob'][ith]).get_data()
        src_sdf = nib.load(source_paths['sdf'][ith]).get_data()
        src_probsdf = nib.load(source_paths['probsdf'][ith]).get_data()

        layout, label_fpfn = {}, {}
        layout['prob'], label_fpfn['prob'] = get_layout_fpfn(src_prob, src_pred, gt, phase='prob')
        layout['probsdf'], label_fpfn['probsdf'] = get_layout_fpfn(src_probsdf, src_pred, gt, phase='probsdf', sdf=src_sdf)

        # save background mask, FG mask
        case_name = (source_paths['gt'][ith].split('/')[-1]).split('_00')[0]
        save_paths = {
            'prob_layout': cmp_uncertain_dir + '%s_prob_layout.nii.gz' % case_name,
            'prob_fpfn': cmp_uncertain_dir + '%s_prob_fpfn.nii.gz' % case_name,
            'probsdf_layout': cmp_uncertain_dir + '%s_probsdf_layout.nii.gz' % case_name,
            'probsdf_fpfn': cmp_uncertain_dir + '%s_probsdf_fpfn.nii.gz' % case_name,
        }
        nib.save(nib.Nifti1Image(layout['prob'], np.eye(4)), save_paths['prob_layout'])
        nib.save(nib.Nifti1Image(label_fpfn['prob'], np.eye(4)), save_paths['prob_fpfn'])
        nib.save(nib.Nifti1Image(layout['probsdf'], np.eye(4)), save_paths['probsdf_layout'])
        nib.save(nib.Nifti1Image(label_fpfn['probsdf'], np.eye(4)), save_paths['probsdf_fpfn'])

if __name__ == '__main__':

    Params = {
        'function':     'metric',    # {'metric', 'filter'}

        # function 'filter' params
        'source':       'prob',      # {'prob', 'probsdf'}
        'probsdf_phase': 'filter',   # {'gen_uncertainty', 'filter'}
    }

    if Params['function'] == 'metric':
        print('Function: Get Metric. Make sure parsing params: --exp_id, --phase, --source')
        calculate_metric()

    elif Params['function'] == 'filter':
        if Params['source'] == 'prob':
            filter_to_pseudo()
        elif Params['source'] == 'probsdf':
            if Params['probsdf_phase'] == 'gen_uncertainty':
                gen_uncertainty_volume()
            else:
                gen_pseudo_with_prob_dist()

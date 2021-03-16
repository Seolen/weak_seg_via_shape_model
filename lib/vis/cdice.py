import torch
import sys
import glob
import os
import nibabel as nib

sys.path.insert(0, os.path.join(sys.path[0], '../../'))
import argparse
import ipdb
import numpy as np

def dice_func(pred, label, smooth=1e-8, label_id=1):
    inter_size = np.sum((((pred == label_id) * (label==label_id)) == 1))
    sum_size = np.sum(pred==label_id) + np.sum(label==label_id)
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice

def calculate_dice(pred_dir, gt_dir):
    paths = {
        'pred': sorted(glob.glob(pred_dir + '*.nii.gz')),
    }
    label_id = 1

    log_path = os.path.join(pred_dir, 'dice_log.txt')
    log = open(log_path, 'w')
    dice_list = []

    length = len(paths['pred'])
    for ith in range(length):
        pred_path = paths['pred'][ith]
        case_name = pred_path.split('/')[-1].split('.')[0]
        if args['label_postfix'] != '':
            gt_path = gt_dir + case_name + '_label.nii.gz'
        else:
            gt_path = gt_dir + case_name + '.nii.gz'
        if use_segthor_gt:
            gt_path = gt_dir + 'Patient_' + case_name[-2:] + '/GT.nii.gz'
            label_id = 3
        pred, gt = nib.load(pred_path).get_data(), nib.load(gt_path).get_data()

        dice = dice_func(pred, gt, label_id=label_id)
        dice_list.append(dice)
        log.write('%s,\t dice: %.4f \n' % (case_name, dice))

    mean, std = np.array(dice_list).mean(), np.array(dice_list).std()
    log.write('\nAvg meter:\ndice: %.4f \n' % (mean))
    log.write('std: %.4f \n' % (std))
    log.close()
    print('mean: %.4f' % mean, 'std: %.4f' % std)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", default='', help="")
    ap.add_argument("--pred_dir", default='', help="")
    ap.add_argument("--gt_dir", default='', help="")
    ap.add_argument("--label_postfix", default='_label', help="")
    args = vars(ap.parse_args())

    use_segthor_gt = False # False
    pred_dirs = [
        '/group/lishl/weak_datasets/0108_SegTHOR/test_pred/1016_trar5_emitw_24/medimg_nii/',

        # '/group/lishl/weak_datasets/LA_dataset/test_pred/1016_lar5_emit_13/medimg_nii/',
        # '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_0.5_label/resample_nii/',
        # '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_0.5_label/crop01_nii/',
        # '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_0.5_label/crop10_nii/',
        # '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_0.5_label/crop11_nii/',
    ]
    # gt_dir = '/group/lishl/weak_datasets/LA_dataset/test/'
    gt_dir = '/group/lishl/weak_datasets/0108_SegTHOR/train/'

    # argparse replace
    if args['pred_dir'] != '':
        pred_dirs = [args['pred_dir']]
    if args['gt_dir'] != '':
        gt_dir = args['gt_dir']
    if args['data_name'] == 'Segthor':
        use_segthor_gt = True      # 1. label id=3 2.label.nii.gz change

    for pred_dir in pred_dirs:
        calculate_dice(pred_dir, gt_dir)
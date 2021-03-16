import os
import nibabel as nib
import numpy as np
from scipy import io
from glob import glob

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import ipdb

def save_pred(data_list, names=None, id=None, title='tmp', phase='val', save_volume=None):
    # save image, gt, preds as png
    # if save_volume=True, save as .nii.gz

    def make_dir(dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    save_dir = os.path.join(os.getcwd(), '../output/%s_%s' % (title, phase))
    make_dir(save_dir)

    for name, data in zip(names, data_list):
        _dir = os.path.join(save_dir, name)
        make_dir(_dir)
        png_prefix = _dir + '/%s' % str(id)

        data = data.numpy()
        if (name in ['pred', 'ae_input', 'ae_pred']):
            data = data.astype(np.uint8)
        B = data.shape[0]

        if save_volume=='3d':
            # .nii.gz
            for ith in range(B):
                save_path = png_prefix + '.nii.gz' #_'%02d.nii.gz' % ith
                output = data[ith] if name != 'image' else data[ith, 0]

                image = nib.Nifti1Image(output, np.eye(4))
                nib.save(image, save_path)



def overlay(title='tmp', phase='val', dataset_2d=False):
    # build overlay image: image + gt + pred
    import cv2
    root_dir = os.path.join(os.getcwd(), '../output/%s_%s' % (title, phase))
    img_dir = os.path.join(root_dir, 'image/')
    pred_dir = os.path.join(root_dir, 'pred/')
    gt_dir = os.path.join(root_dir, 'gt_label/')
    save_dir = os.path.join(root_dir, 'overlay/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_paths = sorted(os.listdir(img_dir))
    pred_paths = sorted(os.listdir(pred_dir))
    gt_paths = sorted(os.listdir(gt_dir))

    if dataset_2d:
        for ith in range(len(img_paths)):
            img = cv2.imread(os.path.join(img_dir, img_paths[ith]))
            label = cv2.imread(os.path.join(pred_dir, pred_paths[ith]))
            label[:, :, :-1] = 0  # pred: red color
            gt = cv2.imread(os.path.join(gt_dir, gt_paths[ith]))
            gt[:, :, 0] = gt[:, :, 2] = 0  # gt: green color (intersection will be yellow)

            label = label + gt
            label[label == 0] = img[label == 0]

            combine = cv2.addWeighted(img, 0.7, label, 0.3, 0)
            cv2.imwrite(os.path.join(save_dir, img_paths[ith]), combine)
    else:
        for filepath in sorted(os.listdir(img_dir)):
            image_path, gt_path, pred_path = img_dir + filepath, gt_dir + filepath, pred_dir + filepath
            image, gt, pred = nib.load(image_path).get_data()[0], nib.load(gt_path).get_data(), nib.load(
                pred_path).get_data()
            for slice_index in range(image.shape[0]):
                png_image, png_gt, png_pred = image[slice_index], gt[slice_index], pred[slice_index]
                save_path = save_dir + '%s_%02d.png' % (filepath.split('.')[0], slice_index)
                output = (png_image - png_image.min()) / ((png_image.max()) - png_image.min()) * 255
                # overlay: img+gt
                out_img = np.repeat(np.expand_dims(output, axis=2), 3, axis=2).astype(np.uint8)
                out_gt = np.zeros_like(out_img, dtype=np.uint8)
                out_gt[:, :, 1] = png_gt * 255  # green
                overlay = cv2.addWeighted(out_img, 0.7, out_gt, 0.3, 0)
                # overlay_2: img+gt+pred
                out_pred = np.zeros_like(out_img, dtype=np.uint8)
                out_pred[:, :, -1] = png_pred * 255  # red
                label = out_gt + out_pred
                overlay_2 = cv2.addWeighted(out_img, 0.7, label, 0.3, 0)

                cv2.imwrite(save_path, overlay_2)

def offline_dice(pred_dir, gt_dir, mode='spfit'):
    import torch
    import sys
    sys.path.insert(0, os.path.join(sys.path[0], '../../'))
    from lib.utils import MultiLossMeter, DiceMetric

    if mode == 'spfit':
        pred_paths = glob(pred_dir + '*_*_*.nii.gz')
        gt_paths = glob(gt_dir + '*.mat')
        log_path = os.path.join(pred_dir, 'dice_log.txt')
        log = open(log_path, 'w')
        metric = DiceMetric(dice_each_class=True)
        dice_meter = MultiLossMeter()
        dice_meter.reset()

        length = len(pred_paths)
        for ith in range(length):
            pred_path, gt_path = pred_paths[ith], gt_paths[ith]
            name = (gt_path.split('/')[-1]).split('.')[0]
            pred, gt = nib.load(pred_path).get_data(), io.loadmat(gt_path)['label']

            pred, gt = torch.from_numpy(np.expand_dims(pred, axis=0).astype(np.float)), \
                       torch.from_numpy(np.expand_dims(gt, axis=0).astype(np.float))
            dices, dice_names = metric.forward(pred, gt)
            dice_meter.update(dices, dice_names)
            log.write('%s,\t dice_fg: %.4f,\t dice_bg: %.4f\n' % (name, dices[0], dices[1]))

        dice_terms = dice_meter.get_metric()
        log.write('\nAvg meter:\ndice_fg: %.4f, dice_bg: %.4f\n' % (dice_terms['dice'], dice_terms['dice_bg']))
        log.close()
        print(dice_terms)

def calculate_fg_weight():
    _dir = '/group/lishl/weak_datasets/Promise12/processed_train_weak_percent_0.5_random/expand_mat/'
          #'/group/lishl/weak_datasets/0108_SegTHOR/processed_trachea_train_weak_percent_0.5_random/expand_mat/'

    data = {'weak_label': [], 'label': []}
    log_path = _dir + 'fg_weight_log.txt'
    log = open(log_path, 'w')

    samples = glob(_dir + '*.mat')
    for sample in samples:
        file_path = sample
        file_data = io.loadmat(file_path)
        for key, value in file_data.items():
            if key in data.keys():
                data[key].append(value)

    weak_weights, gt_weights = [], []
    for ith in range(len(samples)):
        weak, gt = data['weak_label'][ith], data['label'][ith]
        weak_weight = 1.0 * (weak == 0).sum() / (weak == 1).sum()
        gt_weight = 1.0 * (gt == 0).sum() / (gt == 1).sum()
        weak_weights.append(weak_weight)
        gt_weights.append(gt_weight)
        log.write('%s,\t weak_weight: %.1f,\t gt_weight: %.1f\n' % (samples[ith].split('/')[-1], weak_weight, gt_weight))

    log.write('\nAvg weight:\nweak_weight: %.1f, gt_weight: %.1f\n' % (sum(weak_weights)/len(weak_weights),
                                                                       sum(gt_weights)/len(gt_weights)))
    log.close()
    print('weak_weights', sum(weak_weights)/len(weak_weights), '\ngt_weights', sum(gt_weights)/len(gt_weights))

def convert_output_mat(dirname, terms=['image', 'gt_label', 'pred'], save_dirname='mat'):
    save_dir = os.path.join(dirname, '%s/' % save_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    paths = {}
    for term in terms:
        paths[term] = glob(dirname + '%s/*.nii.gz' % term)
    length = len(paths[terms[0]])
    for ith in range(length):
        item = {}
        for term in terms:
            key = 'pred' if 'pseudo' in term else term
            item[key] = nib.load(paths[term][ith]).get_data()
        if '_00' in paths[term][ith]:
            name = ((paths[term][ith]).split('/')[-1]).split('_00')[0]
        else:
            name = ((paths[term][ith]).split('/')[-1]).split('.')[0]
        save_path = os.path.join(save_dir, name+'.mat')
        io.savemat(save_path, item)

def convert_mat2nii(dirname, terms='pred', save_dirname='nii'):
    save_dir = os.path.join(dirname, '%s/' % save_dirname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    paths = sorted(glob(dirname+'*.mat'))
    for path in paths:
        caseid = path.split('/')[-1].split('.mat')[0]
        save_path = save_dir + caseid + '.nii.gz'
        mat = io.loadmat(path)
        pred = mat[terms]
        nib.save(nib.Nifti1Image(pred, np.eye(4)), save_path)


if __name__ == '__main__':
    # convert_mat2nii('/Users/seolen/Seolen-Project/_group/lishl/weak_exp/em_save_pseudo/0929_pror1_emit_21/')
    # calculate_fg_weight()

    # # superpixel fitted out: dice performance
    # pred_dir = '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_val/spfit/'
    # gt_dir = '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_val/mat/'
    # offline_dice(pred_dir, gt_dir, mode='spfit')

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default='', help="exp_id to convert whose pred to mat")
    args = vars(ap.parse_args())


    # 1. Params
    Params = {
        'phase':            'pred',     # {'pred', 'pseudo'}

        'pseudo_phase':     'prob'      # {'prob', 'probsdf'}
    }
    dirs = [
        # '1012_pro_r1_02_train/', '1012_pro_r1_02_val/',
        # '1012_pro_r3_02_train/', '1012_pro_r3_02_val/',
        # '1012_pro_r5_02_train/', '1012_pro_r5_02_val/',
        # '1012_tra_r1_01_train/', '1012_tra_r1_01_val/',
        # '1012_tra_r3_01_train/', '1012_tra_r3_01_val/',
        # '1012_tra_r5_01_train/', '1012_tra_r5_01_val/',
        # '1012_la_r1_01_train/', '1012_la_r1_01_val/',
        # '1012_la_r3_01_train/', '1012_la_r3_01_val/',
        # '1012_la_r5_01_train/', '1012_la_r5_01_val/',
        # '1012_la_r10_01_train/', '1012_la_r10_01_val/',

        # 'k1012_pro_r1_01_train/', 'k1012_pro_r1_01_val/',
        # 'k1012_pro_r3_01_train/', 'k1012_pro_r3_01_val/',
        # 'k1012_pro_r5_01_train/', 'k1012_pro_r5_01_val/',
        # 'k1012_tra_r1_01_train/', 'k1012_tra_r1_01_val/',
        # 'k1012_tra_r3_01_train/', 'k1012_tra_r3_01_val/',
        # 'k1012_tra_r5_01_train/', 'k1012_tra_r5_01_val/',
        # 'k1012_la_r1_01_train/', 'k1012_la_r1_01_val/',
        # 'k1012_la_r3_01_train/', 'k1012_la_r3_01_val/',
        # 'k1012_la_r5_01_train/', 'k1012_la_r5_01_val/',

        # '1012_la_r10_01_train/', '1012_la_r10_01_val/',
        # '1012_pro_r10_01_train/', '1012_pro_r10_01_val/',
        # '1012_tra_r10_02_train/', '1012_tra_r10_02_val/',

        # '1103_trar3_emitu_01_train/', '1103_trar3_emitu_01_val/',

        # 'k1012_tra_far_r1_01_train/', 'k1012_tra_far_r1_01_val/',
        # 'k1012_tra_far_r3_01_train/', 'k1012_tra_far_r3_01_val/',
        # 'k1012_tra_far_r5_01_train/', 'k1012_tra_far_r5_01_val/',

        # 's1012_tra_r1_01_train/', 's1012_tra_r1_01_val/',
        # 's1012_tra_r3_01_train/', 's1012_tra_r3_01_val/',
        # 's1012_tra_r5_01_train/', 's1012_tra_r5_01_val/',
        # 't1012_tra_r1_01_train/', 't1012_tra_r1_01_val/',
        # 't1012_tra_r3_01_train/', 't1012_tra_r3_01_val/',
        # 't1012_tra_r5_01_train/', 't1012_tra_r5_01_val/',

        # 't0312_tra_r1_01_train/', 't0312_tra_r3_01_train/', 't0312_tra_r5_01_train/',    # note: +/
        # 't0312_tra_r1_01_val/', 't0312_tra_r3_01_val/', 't0312_tra_r5_01_val/',
        # 't0312_tra_r10_01_train/', 't0312_tra_r10_01_val/',
        't0312_pro_r1_02_train/', 't0312_pro_r3_02_train/', 't0312_pro_r5_02_train/',    # note: +/
        't0312_pro_r1_02_val/', 't0312_pro_r3_02_val/', 't0312_pro_r5_02_val/',
        't0312_pro_r10_02_train/', 't0312_pro_r10_02_val/',
    ]

    # 2. Convert .nii.gz to .mat
    prefix = '/group/lishl/weak_exp/output/'
    # if parse args from outer cmd, replace dirs.
    if args['exp_id'] != '':
        dirs = [args['exp_id'] + '_train/', args['exp_id'] + '_val/',]

    dirs = [prefix + _dir for _dir in dirs]
    if Params['phase'] == 'pred':
        for _dir in dirs:
            convert_output_mat(_dir, terms=['image', 'gt_label', 'pred'])
    elif Params['phase'] == 'pseudo':
        if Params['pseudo_phase'] == 'prob':
            term_list = ['pseudo_0.1_2.0', 'pseudo_0.2_2.0', 'pseudo_0.3_2.0', 'pseudo_0.4_2.0', 'pseudo_0.5_2.0']
        elif Params['pseudo_phase'] == 'probsdf':
            term_list = ['pseudo_probsdf_0.1_2.0', 'pseudo_probsdf_0.2_2.0', 'pseudo_probsdf_0.3_2.0',
                         'pseudo_probsdf_0.4_2.0', 'pseudo_probsdf_0.5_2.0']
        for _dir in dirs:
            for term in term_list:
                convert_output_mat(_dir, terms=[term], save_dirname='mat_pseudo/%s' % term)





    '''
    prefix = '/group/lishl/weak_exp/output/'
    # dirs = [
    #     '0825_trachea_r3_01_train/',
    #     '0825_trachea_r1_01_train/', '0825_trachea_r5_01_train/',
    #     '0825_pro_balance_r1_01_train/', '0825_pro_balance_r3_01_train/', '0825_pro_balance_r5_01_train/'
    # ]
    # term_list = ['pseudo_probsdf_0.1_2.0', 'pseudo_probsdf_0.2_2.0', 'pseudo_probsdf_0.3_2.0',
    #              'pseudo_probsdf_0.4_2.0', 'pseudo_probsdf_0.5_2.0']

    dirs = [
        # '0907_pro_r1_em21_filter40_train/', '0907_trachea_r1_em21_filter50_train/', '0907_trachea_r3_em21_filter50_train/',
        # '0908_pro_r1_em21_filter50_train/', '0908_trachea_r1_em21_filter50_train/',
        '0908_trachea_r3_em21_filter50_train/',
    ]
    term_list = ['pseudo_0.1_2.0', 'pseudo_0.2_2.0', 'pseudo_0.3_2.0', 'pseudo_0.4_2.0', 'pseudo_0.5_2.0']

    dirs = [prefix + _dir for _dir in dirs]
    for _dir in dirs:
        for term in term_list:
            convert_output_mat(_dir, terms=[term], save_dirname='mat_pseudo/%s'%term)
    '''
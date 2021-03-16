import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import random
from pathlib import Path
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
import skimage.io as io
import numpy as np
import copy
import datetime
from tqdm import tqdm
import math
import scipy.io
import json
import ipdb
from skimage.measure import regionprops

from glob import glob
from data_process.process_pipeline2 import make_save_dirs, compute_final_spacing, save_params, get_la_paths
random.seed(0)


def collect_params(general_dir, cor_dir):
    Cparams = {
        'data_name':        Params['data_name'],       # {'Promise', 'Segthor'}
        'n_percent':        Params['n_percent'],
        'cor_dir':          cor_dir,       # corresponding train data dir
        
        'original_ss':      None,       # original shape and spacing of each case
        'target_spacing':   None,
        'max_shape':        None,       # Align to center: max shape
        'zoom_factor':      None,       # Crop: downsample zoom factor
        'roi':              None,       # Crop: roi to crop
        'down_n_all':       None,

        # 'resampled_shape':  None,
        # 'downscaled_shape': None,
    }

    # 1. load original_ss from general_dir
    with open(general_dir + 'original.json', 'r') as f:
        Cparams['original_ss'] = json.load(f)

    # 2. load others
    with open(cor_dir + 'splits/params.json') as f:
        params = json.load(f)
        Cparams['target_spacing'] = params['re_sample']
        Cparams['zoom_factor'] = params['down_sample_ratios']
        Cparams['roi'] = params['roi']
        Cparams['down_n_all'] = params['down_n_all']

    # 3. max shape
    path_max_shape = cor_dir + 'align_nii/Case01_label.nii.gz'
    max_shape = nib.load(path_max_shape).get_data().shape
    Cparams['max_shape'] = max_shape

    return Cparams

################################################################################################
################################################################################################
################################################################################################

def save_case(step, case_name, image, data, save_nii_dir, save_mat_dir):
    image_nib = nib.Nifti1Image(copy.deepcopy(image), np.eye(4))
    nib.save(image_nib, os.path.join(save_nii_dir, '{}.nii.gz'.format(case_name)))
    if step in ['normalize', 'expand', 'graphcut']:
        scipy.io.savemat(os.path.join(save_mat_dir, '{}.mat'.format(case_name)), data[case_name])
    scipy.io.savemat(os.path.join(out_path, step + '.mat'), data)

def load_data(step, other_load_path=None):
    data_dir = os.path.join(out_path, step + '.mat')
    if other_load_path is not None:
        data_dir = os.path.join(other_load_path, step + '.mat')
    data = scipy.io.loadmat(data_dir)
    data_new = {}
    for case_name in data.keys():
        if 'Case' not in case_name:
            continue
        data_new[case_name] = {
            'image': data[case_name]['image'][0, 0],
            'spacing': data[case_name]['spacing'][0, 0]
        }
    return data_new

def get_nii_paths(root, phase):
    if Params['data_name'] == 'Segthor':
        data_dir = root + '/%s/' % phase
        image_paths = sorted(glob(data_dir + '*.nii.gz'))
    elif Params['data_name'] == 'LA':
        data_dir = root + '/%s/' % phase
        image_paths = sorted(glob(data_dir + '*.nii.gz'))
        image_paths = [image_path for image_path in image_paths if '_label.nii.gz' not in image_path]
    return image_paths

def load_nii(root, phase, out_path, depth_transpose=False):
    step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    paths = get_nii_paths(root, phase)
    data = {}

    for image_p in tqdm(sorted(paths)):
        if Params['data_name'] == 'Segthor':
            case_id = image_p.split('/')[-1][8:10]
            case_name = 'Case%02d' % (int(case_id))
        elif Params['data_name'] == 'LA':
            case_name = image_p.split('/')[-1][:6]

        nii_image = sitk.ReadImage(image_p)
        spacing = nii_image.GetSpacing()
        image = nib.load(image_p).get_data()
        if depth_transpose:
            image = np.transpose(image, (2, 0, 1))
            spacing = (spacing[2], spacing[0], spacing[1])
        data[case_name] = {
            'image': image,
            'spacing': spacing
        }

        print('load  ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print(case_name, spacing)
        print(image_p)
        save_case(step, case_name, image, data, save_nii_dir, save_mat_dir)

def resample(out_path):
    step = 'resample'
    prev_step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        spacing = data[case_name]['spacing'][0]
        print(case_name)
        print('before', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        if max(spacing) / min(spacing) > 3:
            anisotropic = True
        else:
            anisotropic = False
        if re_sample[0] != 0 or re_sample[1] != 0 or re_sample[2] != 0:
            zoom_factor = [spacing[i] / re_sample[i] for i in range(3)]
            if zoom_factor != [1,1,1]:
                if anisotropic:
                    orders = [3, 1, 0, 0]
                else:
                    orders = [3, 1, 3, 1]
                assert np.argmax(spacing) == 0  # z axis
                image = ndimage.zoom(image, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[0])
                image = ndimage.zoom(image, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[2])
        data[case_name]['image'] = image

        print('after ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print(case_name, spacing)
        save_case(step, case_name, image, data, save_nii_dir, save_mat_dir)

def align_volume_center(out_path):
    step = 'align'
    prev_step = 'resample'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    max_shape = Cparams['max_shape']
    max_center = np.array(max_shape) // 2
    for case_name in data.keys():
        image = data[case_name]['image']
        image_shape = np.array(image.shape)
        image_center = image_shape // 2
        start = (max_center - image_center).astype(int)
        end = (start + image_shape).astype(int)

        print(case_name)
        print('before', image.shape, type(image[0, 0, 0]), image.max(), image.sum())
        image_new = np.zeros(max_shape).astype(np.int16)
        image_new[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = copy.deepcopy(image)
        image = image_new
        data[case_name]['image'] = image

        print('after ', image.shape, type(image[0, 0, 0]), image.max(), image.sum())
        save_case(step, case_name, image, data, save_nii_dir, save_mat_dir)

def crop_roi(out_path, changed_axis=2):
    step = 'crop'
    prev_step = 'align'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    zoom_factor, roi, down_n_all = Cparams['zoom_factor'], Cparams['roi'], Cparams['down_n_all']
    for case_name in data.keys():
        image = data[case_name]['image']

        print(case_name)
        print('before    ', image.shape, type(image[0, 0, 0]), image.max(), image.sum())

        orders = [3, 1, 3, 1]
        image = ndimage.zoom(image, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[0])
        # downsample along Z axis
        if changed_axis != 2:
            image = ndimage.zoom(image, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[2])
        print('after down', image.shape, type(image[0, 0, 0]), image.max(), image.sum())

        if Params['data_name'] == 'LA' and roi[1]-roi[0]>image.shape[0]:
            new_shape = (roi[1] - roi[0], roi[3]-roi[2], roi[5]-roi[4])
            image_new = np.zeros(new_shape).astype(np.int16)
            zlen = roi[1]-roi[0]; z_start = (zlen-image.shape[0])//2
            image_new[z_start:z_start+image.shape[0], :, :] = image[:, roi[2]: roi[3], roi[4]: roi[5]]
            image = image_new
        else:
            image = image[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        data[case_name]['image'] = image
        print('after crop', image.shape, type(image[0, 0, 0]), image.max(), image.sum())

        save_case(step, case_name, image, data, save_nii_dir, save_mat_dir)
    return zoom_factor, roi, down_n_all

def normalize_intensity(out_path, data_name='Promise'):
    step = 'normalize'
    prev_step = 'crop'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']

        print(case_name)
        print('before', image.shape, type(image[0, 0, 0]), image.max(), image.sum(), image.min())

        if data_name in ['Promise', 'promise']:
            image_value = image[image != 0]
            mean = image_value.mean()
            std = image_value.std()
            image = (image - mean) / std
            data[case_name]['image'] = image
        elif data_name in ['Segthor', 'segthor']:
            clip_window = [-986, 271]
            mean, std = 20.78, 180.50
            image[image < clip_window[0]] = clip_window[0]
            image[image > clip_window[1]] = clip_window[1]
            image = (image - mean) / std
            data[case_name]['image'] = image
        elif data_name in ['LA']:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / std
            # data[case_name]['image'] = image  # We forget this re-assignment in current exps, yet it doesn't bring hurt anayway.

        print('mean std', mean, std, image.shape, image.shape[0]*image.shape[1]*image.shape[2])
        print('after ', image.shape, type(image[0, 0, 0]), image.max(), image.min())
        save_case(step, case_name, image, data, save_nii_dir, save_mat_dir)

########################################### TODO: Recover Process ###########################################
def load_data_label(step, other_load_path=None):
    data_dir = os.path.join(out_path, step + '.mat')
    if other_load_path is not None:
        data_dir = os.path.join(other_load_path, step + '.mat')
    data = scipy.io.loadmat(data_dir)
    data_new = {}
    for case_name in data.keys():
        if 'Case' not in case_name:
            continue
        data_new[case_name] = {
            'label': data[case_name]['label'][0, 0],
        }
    return data_new

def save_case_label(step, case_name, label, data, save_nii_dir, save_mat_dir):
    if step in ['medimg'] or print_verbose:
        image_nib = nib.Nifti1Image(copy.deepcopy(label), np.eye(4))
        nib.save(image_nib, os.path.join(save_nii_dir, '{}.nii.gz'.format(case_name)))
    if step in ['normalize', 'expand', 'graphcut']:
        scipy.io.savemat(os.path.join(save_mat_dir, '{}.mat'.format(case_name)), data[case_name])
    scipy.io.savemat(os.path.join(out_path, step + '.mat'), data)

def load_pred(load_dir, out_path, phase):
    step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)

    # load labels from given dir (record their ids)
    paths = sorted(glob(load_dir + '*.nii.gz'))
    data = {}
    for image_p in sorted(paths):
        case_name = image_p.split('/')[-1][:6]
        label = nib.load(image_p).get_data()

        data[case_name] = {
            'label': label,
        }
        if print_verbose:
            print('load  ', label.shape, type(label[0, 0, 0]), label.max(), label.min())
            print(case_name, image_p)
        save_case_label(step, case_name, label, data, save_nii_dir, save_mat_dir)

def reverse_crop(out_path):
    step = 'crop'
    prev_step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data_label(prev_step)

    zoom_factor, roi, down_n_all = Cparams['zoom_factor'], Cparams['roi'], Cparams['down_n_all']
    # calculate downsampled max_shape
    max_shape = Cparams['max_shape']
    downsampled_max_shape = [int(round(ia * ib)) for ia, ib in zip(max_shape, zoom_factor)]

    for case_name in data.keys():
        label = data[case_name]['label']
        if print_verbose:
            print(case_name)
            print('before    ', label.shape, type(label[0, 0, 0]), label.max(), label.min())
        # de-crop: original unified shape
        label_new = np.zeros(downsampled_max_shape).astype(np.uint8)
        if Params['data_name'] == 'LA':
            zlen = roi[1]-roi[0]; roi_start = (zlen-label_new.shape[0])//2
            label_new[:, roi[2]: roi[3], roi[4]: roi[5]] = label[roi_start: roi_start+label_new.shape[0], :, :]
        else:
            label_new[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]] = label.copy()
        label = label_new
        if print_verbose:
            print('after de-crop', label.shape, type(label[0, 0, 0]), label.max(), label.min())

        # de-downsample: original max shape
        label_new = np.zeros(max_shape).astype(np.uint8)
        orders = [3, 1, 3, 1]
        label = ndimage.zoom(label, zoom=[1.0, 1/zoom_factor[1], 1/zoom_factor[2]], order=orders[1])
        if zoom_factor[0] != 1:
            label = ndimage.zoom(label, zoom=[1/zoom_factor[0], 1.0, 1.0], order=orders[3])

        if label_new.shape != label.shape:
            if label.shape[1] > label_new.shape[1] and label.shape[2] > label_new.shape[2]:     # for trachea
                # label_new = label[:, :label_new.shape[1], :label_new.shape[2]]    #00
                # label_new = label[:, :label_new.shape[1], 1:]   # 01
                # label_new = label[:, 1:, :label_new.shape[1]]  # 10
                _t1, _t2 = label.shape[1] - label_new.shape[1], label.shape[2] - label_new.shape[2]
                label_new = label[:, _t1:, _t2:]  # for trachea, we find this way better  # 11
            elif label.shape[1] < label_new.shape[1] and label.shape[2] < label_new.shape[2]:   # for prostate
                # label_new[:, :label.shape[1], :label.shape[2]] = label      # 00
                # label_new[:, :label.shape[1], 1:] = label   # 01
                # label_new[:, 1:, :label.shape[1]] = label  # 10
                _t1, _t2 = label_new.shape[1] - label.shape[1], label_new.shape[2] - label.shape[2]
                label_new[:, 1:, 1:] = label  # 11
            else:
                ipdb.set_trace()
                print(label_new.shape, label.shape)
                raise NotImplementedError
            label = label_new
        if print_verbose:
            print('after de-downsample', label.shape, type(label[0, 0, 0]), label.max(), label.min())
        data[case_name]['label'] = label
        save_case_label(step, case_name, label, data, save_nii_dir, save_mat_dir)

def reverse_align(out_path):
    step = 'align'
    prev_step = 'crop'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data_label(prev_step)

    # check max_shape consistency
    max_shape = Cparams['max_shape']
    original_ss = Cparams['original_ss']
    target_spacing = Cparams['target_spacing']

    for case_name in data.keys():
        label = data[case_name]['label']
        label_shape = np.array(label.shape)
        label_center = label_shape // 2
        if print_verbose:
            print(case_name)

        # calculate resampled shape
        original_spacing, original_shape = original_ss[case_name]['spacing'], original_ss[case_name]['shape']
        resampled_shape = [int(round(original_shape[ith] * original_spacing[ith] / target_spacing[ith])) for ith in range(len(original_shape))]
        original_center = np.array(resampled_shape) // 2
        start = (label_center - original_center).astype(int)
        end = (start + resampled_shape).astype(int)
        if print_verbose:
            print('before    ', label.shape, type(label[0, 0, 0]), label.max(), label.min())

        # de-align: depadding to separate shape (after resample)
        label_new = label[start[0]:end[0], start[1]:end[1], start[2]:end[2]].copy()
        label = label_new

        data[case_name]['label'] = label
        if print_verbose:
            print('after reverse-align', label.shape, type(label[0, 0, 0]), label.max(), label.min())
        save_case_label(step, case_name, label, data, save_nii_dir, save_mat_dir)

def reverse_resample(out_path):
    step = 'resample'
    prev_step = 'align'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data_label(prev_step)

    original_ss = Cparams['original_ss']

    # de-resample: resampled unified spacing to original spacings (or according to shape ratio)
    for case_name in data.keys():
        label = data[case_name]['label']
        label_shape = label.shape
        original_shape = original_ss[case_name]['shape']
        zoom_factor = [original_shape[i] / label_shape[i] for i in range(3)]
        if print_verbose:
            print(case_name)
            print('before    ', label.shape, type(label[0, 0, 0]), label.max(), label.min())
        orders = [3, 1, 3, 1]
        if zoom_factor != [1,1,1]:
            label = ndimage.zoom(label, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
            label = ndimage.zoom(label, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])

        data[case_name]['label'] = label
        assert sorted(label.shape) == sorted(original_shape)
        if print_verbose:
            print('after de-resample', label.shape, type(label[0, 0, 0]), label.max(), label.min())
        save_case_label(step, case_name, label, data, save_nii_dir, save_mat_dir)

def recover_medimg(out_path):
    step = 'medimg'
    prev_step = 'resample'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data_label(prev_step)

    original_ss = Cparams['original_ss']

    # de-resample: resampled unified spacing to original spacings (or according to shape ratio)
    for case_name in data.keys():
        label = data[case_name]['label']
        label_shape = label.shape
        if print_verbose:
            print(case_name)

        # recover original label id (3 for trachea)
        if Params['data_name'] == 'Segthor':
            label[label == 1] = 3   # trachea label id

        # recover XYZ order
        original_spacing = original_ss[case_name]['spacing'].copy()
        if Params['depth_transpose']:
            label = np.transpose(label, (1, 2, 0))
            original_spacing = [original_spacing[1], original_spacing[2], original_spacing[0]]
        if print_verbose:
            print('after recover medimg', label.shape, original_spacing)
        # equip label file with original spacings (spacing setting)
        original_spacing.append(1)
        image_nib = nib.Nifti1Image(label, np.diag(original_spacing))
        nib.save(image_nib, os.path.join(save_nii_dir, '{}.nii.gz'.format(case_name)))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_name", default='', help="")
    ap.add_argument("--n_percent", default=0.01, type=float, help="")
    ap.add_argument("--recover_id", default='', help="")
    ap.add_argument("--dirprefix", default='', help="")
    ap.add_argument("--phase", default='', help="")
    args = vars(ap.parse_args())

    Params = {
        'data_name':        'Segthor',      # {'Promise', 'Segthor', 'LA'}    当前不做Promise系列test
        'recover_label':    True,         # {True: recover pred label to original shape; False: process test input}
        'print_verbose':    False,      # print intermediate info or not

        'n_percent':        1.0,            # {0.5, 0.3, 0.1}
        'dirprefix':        '1011_',
        'phase':            'test',
        'depth_transpose':  True,           # True for Segthor

        'recover_id':       '1016_trar5_emitw_24'  # '1016_lar5_emit_13'
        # 'recover_label_dir': '/group/lishl/weak_exp/output/1016_lar5_emit_13_test/ae_pred/', # pred label dir (.nii)
        # '/group/lishl/weak_exp/output/1016_trar5_emitw_24_val/pred/'
    }

    # argparse replace
    for keyword in ['data_name', 'n_percent', 'recover_id', 'dirprefix', 'phase']:
        if (keyword not in ['n_percent', 'print_verbose']) and args[keyword] != '':
            Params[keyword] = args[keyword]
        elif keyword == 'n_percent' and args[keyword] != 0.01:
            Params[keyword] = args[keyword]

    root_dirs = {'Segthor': '/group/lishl/weak_datasets/0108_SegTHOR', 'LA': '/group/lishl/weak_datasets/LA_dataset'}
    root = root_dirs[Params['data_name']]
    if Params['recover_label']:
        Params['recover_label_dir'] = '/group/lishl/weak_exp/output/%s_test/pred/' % Params['recover_id'] # test, val, ae_pred, pred
        out_path = os.path.join(root, 'test_pred/%s' % Params['recover_id'])
        print('Saving to ', out_path)
    else:
        out_path = os.path.join(root, Params['dirprefix'] + Params['phase'] + '_weak_percent_{}'.format(Params['n_percent']))
    print_verbose = Params['print_verbose']
    # if Params['recover_label']:
    #      out_path = out_path + '_label'

    Dirs = {
        'Segthor':{
            'general': '/group/lishl/weak_datasets/0108_SegTHOR/test/',
            # 'general': '/group/lishl/weak_datasets/0108_SegTHOR/1011_trachea_train_weak_random/',
            1.0: '/group/lishl/weak_datasets/0108_SegTHOR/%strachea_train_weak_percent_1.0_random/' % Params['dirprefix'],
            0.5: '/group/lishl/weak_datasets/0108_SegTHOR/%strachea_train_weak_percent_0.5_random/' % Params['dirprefix'],
            0.3: '/group/lishl/weak_datasets/0108_SegTHOR/%strachea_train_weak_percent_0.3_random/' % Params['dirprefix'],
            0.1: '/group/lishl/weak_datasets/0108_SegTHOR/%strachea_train_weak_percent_0.1_random/' % Params['dirprefix'],
        },
        'LA':{
            'general': '/group/lishl/weak_datasets/LA_dataset/test/',
            1.0: '/group/lishl/weak_datasets/LA_dataset/%strain_weak_percent_1.0_random/' % Params['dirprefix'],
            0.5: '/group/lishl/weak_datasets/LA_dataset/%strain_weak_percent_0.5_random/' % Params['dirprefix'],
            0.3: '/group/lishl/weak_datasets/LA_dataset/%strain_weak_percent_0.3_random/' % Params['dirprefix'],
            0.1: '/group/lishl/weak_datasets/LA_dataset/%strain_weak_percent_0.1_random/' % Params['dirprefix'],
        },
    }
    general_dir, cor_dir = Dirs[Params['data_name']]['general'], Dirs[Params['data_name']][Params['n_percent']]
    Cparams = collect_params(general_dir, cor_dir)
    re_sample = Cparams['target_spacing']
    changed_axis = 2

    if not Params['recover_label']:     # process test images to model input
        # 0. load test data
        load_nii(root, Params['phase'], out_path, depth_transpose=Params['depth_transpose'])
        resample(out_path)
        align_volume_center(out_path)
        zoom_factor, roi, down_n_all = crop_roi(out_path, changed_axis=changed_axis)
        normalize_intensity(out_path, data_name=Cparams['data_name'])

        final_spacing = compute_final_spacing(re_sample, zoom_factor)
        save_params({
            'datetime': str(datetime.datetime.now()),
            'n_percent': Cparams['n_percent'],
            're_sample': re_sample,
            'out_path': out_path,
            'down_sample_ratios': zoom_factor,
            'down_n_all': down_n_all,
            'final_spacing': final_spacing,
            'roi': roi,
            'crop_shape': [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
        }, out_path)

    else:       # recover predicted label to original shape and spacing.
        # Recover original label
        load_pred(Params['recover_label_dir'], out_path, Params['phase'])
        reverse_crop(out_path)
        reverse_align(out_path)
        reverse_resample(out_path)
        recover_medimg(out_path)
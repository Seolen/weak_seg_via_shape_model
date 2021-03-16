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
import skimage
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
from skimage.io import imread, imsave
from skimage.transform import resize
from numpy import unique as uniq
import warnings
from functools import partial
from typing import Any, Callable, List, Tuple
from PIL import Image
import nrrd


random.seed(0)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_input_paths(root, phase):
    in_data_dir = os.path.join(root, phase)
    in_data_dir = Path(in_data_dir)
    nii_paths = [str(p) for p in in_data_dir.rglob('*.mhd')]
    img_nii_paths = sorted(p for p in nii_paths if "_segmentation" not in str(p))
    gt_nii_paths = sorted(p for p in nii_paths if "_segmentation" in str(p))
    paths = list(zip(img_nii_paths, gt_nii_paths))
    return paths

def get_nii_paths(root, phase):
    image_paths, label_paths = [], []
    data_dir = root + '/train/'
    for subdir in sorted(os.listdir(data_dir)):
        if subdir == '.DS_Store':
            continue
        image_paths.append(os.path.join(data_dir, os.path.join(subdir, subdir+'.nii.gz')))
        label_paths.append(os.path.join(data_dir, os.path.join(subdir, 'GT.nii.gz')))
    paths = list(zip(image_paths, label_paths))
    return paths

def get_la_paths(root, phase):
    image_paths, label_paths = [], []
    data_dir = root + '/2018LA_Seg_Training Set/'
    subdirs = []
    with open(os.path.join(root, '%s.list' % phase)) as f:
        for line in f:
            subdir = line.strip()
            if subdir != '':
                subdirs.append(subdir)
    for subdir in subdirs:
        if subdir == '.DS_Store':
            continue
        image_paths.append(os.path.join(data_dir, os.path.join(subdir, 'lgemri.nrrd')))
        label_paths.append(os.path.join(data_dir, os.path.join(subdir, 'laendo.nrrd')))
    paths = list(zip(image_paths, label_paths))
    return paths


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
            'label': data[case_name]['label'][0, 0],
            'weak_label': data[case_name]['weak_label'][0, 0],
            'slice_ids': data[case_name]['slice_ids'][0, 0],
            'num_slices': data[case_name]['num_slices'][0, 0],
            'spacing': data[case_name]['spacing'][0, 0]
        }
    return data_new


def make_save_dirs(step, out_path):
    save_nii_dir = os.path.join(out_path, step + '_nii')
    if not os.path.exists(save_nii_dir):
        os.makedirs(save_nii_dir)
    save_mat_dir = os.path.join(out_path, step + '_mat')
    if step in ['normalize', 'expand', 'expand_bg', 'graphcut']:
        if not os.path.exists(save_mat_dir):
            os.makedirs(save_mat_dir)
    return save_nii_dir, save_mat_dir


def save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir):
    image_nib = nib.Nifti1Image(copy.deepcopy(image), np.eye(4))
    label_nib = nib.Nifti1Image(copy.deepcopy(label).astype(np.uint8), np.eye(4))
    weak_label_nib = nib.Nifti1Image(copy.deepcopy(weak_label).astype(np.uint8), np.eye(4))
    nib.save(image_nib, os.path.join(save_nii_dir, '{}.nii.gz'.format(case_name)))
    nib.save(label_nib, os.path.join(save_nii_dir, '{}_label.nii.gz'.format(case_name)))
    nib.save(weak_label_nib, os.path.join(save_nii_dir, '{}_weak_label.nii.gz'.format(case_name)))
    if step in ['normalize', 'expand', 'expand_bg', 'graphcut']:
        scipy.io.savemat(os.path.join(save_mat_dir, '{}.mat'.format(case_name)), data[case_name])
    scipy.io.savemat(os.path.join(out_path, step + '.mat'), data)


def gen_two_lines(arr, line_width, boundary_margin):
    # arr: 2D image
    # method: find center -> generate two lines -> filter irrelavant region -> return two lines

    def find_center(arr):
        (X, Y) = arr.shape
        arr_mask = (arr > 0).astype(np.float32)
        arr_mask = arr_mask / arr_mask.sum()

        dx = np.sum(arr_mask, 1)
        dy = np.sum(arr_mask, 0)
        # expected values
        cx = np.sum(dx * np.arange(X))
        cy = np.sum(dy * np.arange(Y))
        return int(cx), int(cy)

    if (arr==1).sum() == 0:
        return arr.copy()

    # one random region if multiple fg regions
    labeled_arr, num_features = ndimage.label(arr)
    if num_features > 1:
        selected_cc = random.choice(list(range(1, num_features+1)))
        arr = labeled_arr == selected_cc

    # find center
    center = find_center(arr)

    # two lines
    weak_label = (arr.copy() > 0).astype(np.uint8)
    shift = line_width // 2
    weak_label[center[0] - shift:center[0] - shift + line_width, :] = 2
    if not (to_kernelcut or to_scribble):
        weak_label[:, center[1] - shift:center[1] - shift + line_width] = 2

    # filter irrelavant region
    weak_label[arr == 0] = 0
    weak_ref = weak_label.copy()
    # nonzero = np.nonzero(weak_label == 2)
    nonzero = np.nonzero(arr==1)
    x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
    p_margin = boundary_margin
    while True:
        if not to_kernelcut:
            weak_label[x0:x0 + p_margin, center[1] - shift:center[1] - shift + line_width] \
                = weak_label[x1 - p_margin + 1:x1 + 1, center[1] - shift:center[1] - shift + line_width] = 1
            weak_label[center[0] - shift:center[0] - shift + line_width, y0:y0 + p_margin] \
                = weak_label[center[0] - shift:center[0] - shift + line_width, y1 - p_margin + 1:y1 + 1] = 1
        else:
            weak_label[:, :y0 + p_margin] = 1
            weak_label[:, y1 - p_margin] = 1

        if (weak_label == 2).sum() > 0:
            break
        else:
            p_margin = p_margin // 2
            weak_label = weak_ref.copy()

    # value: 0 unlabeled, 1 labeled
    weak_label[weak_label < 2] = 0
    weak_label[weak_label == 2] = 1

    return weak_label

def gen_long_scribble(arr, line_width, boundary_margin):
    # arr: 2D image
    # method: find center -> generate 6 lines -> select the longest one
    #                     -> filter irrelevant region -> return two lines

    def find_center(arr):
        (X, Y) = arr.shape
        arr_mask = (arr > 0).astype(np.float32)
        arr_mask = arr_mask / arr_mask.sum()

        dx = np.sum(arr_mask, 1)
        dy = np.sum(arr_mask, 0)
        # expected values
        cx = np.sum(dx * np.arange(X))
        cy = np.sum(dy * np.arange(Y))
        return int(cx), int(cy)
    def cal_tan_cot(rad, default=10000):
        tan1, cot1 = 0, 0
        try:
            tan1 = math.tan(rad)
        except:
            tan1 = default
        try:
            cot1 = 1 / math.tan(rad)
        except:
            cot1 = default
        return tan1, cot1

    if (arr==1).sum() == 0:
        return arr.copy()

    # one random region if multiple fg regions
    labeled_arr, num_features = ndimage.label(arr)
    if num_features > 1:
        selected_cc = random.choice(list(range(1, num_features+1)))
        arr = labeled_arr == selected_cc

    # find center
    center = find_center(arr)

    # major axis and orientation
    shift = line_width // 2
    weak_label = (arr.copy() > 0).astype(np.uint8)
    prop = regionprops(weak_label)[0]
    orientation = prop.orientation
    candidates = [0, math.pi/6, math.pi/3, math.pi/2, -math.pi/6, -math.pi/3, -math.pi/2]
    rad = candidates[min(range(len(candidates)), key=lambda i: abs(candidates[i]-orientation))]
    x0, y0 = center
    bbox = prop.bbox
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # get tan() and cot()
    tan1, cot1 = cal_tan_cot(rad)

    # generate major axis
    current = 2     # label id
    if rad == 0:
        weak_label[:, y0 - shift:y0 - shift + line_width] = current
    elif rad == math.pi/2 or rad == -math.pi/2:  # vertical line
        weak_label[x0 - shift: x0 - shift +line_width, :] = current
    else:
        for x_gap in range(1, x0 - xmin):
            xt = x0 - x_gap
            yt = int(y0 - x_gap * tan1)
            if yt >= ymin and yt <= ymax:
                weak_label[xt, yt] = current
                # line_width
                weak_label[xt-shift:xt-shift+line_width, yt-shift:yt-shift+line_width] = current
        for x_gap in range(1, xmax - x0):
            xt = x0 + x_gap
            yt = int(y0 + x_gap * tan1)
            if yt >= ymin and yt <= ymax:
                weak_label[xt, yt] = current
                # line_width
                weak_label[xt - shift:xt - shift + line_width, yt-shift:yt-shift+line_width] = current

    # filter irrelavant region
    weak_label[arr == 0] = 0
    weak_ref = weak_label.copy()
    p_margin = boundary_margin
    while True:
        # erode original mask, and filter out irrelevant region
        arr_eroded = ndimage.binary_erosion(arr, iterations=p_margin).astype(arr.dtype)
        weak_label[arr_eroded == 0] = 1
        if (weak_label==2).sum() > 0:
            break
        p_margin = p_margin // 2
        weak_label = weak_ref.copy()
        if p_margin == 0:
            break

    # check one longest CC
    labeled_arr, num_features = ndimage.label(weak_label == 2)
    if num_features > 1:
        largestCC = labeled_arr == np.argmax(np.bincount(labeled_arr.flat)[1:]) + 1
        weak_label[np.logical_and(largestCC != 1, weak_label == 2)] = 1

    # value: 0 unlabeled, 1 labeled
    weak_label[weak_label < 2] = 0
    weak_label[weak_label == 2] = 1
    return weak_label

def gen_point_label(arr, line_width, boundary_margin):
    '''
    FG label: random point, with radius=line_width
    '''
    # 0. one random region if multiple fg regions
    labeled_arr, num_features = ndimage.label(arr)
    if num_features > 1:
        selected_cc = random.choice(list(range(1, num_features + 1)))
        arr = labeled_arr == selected_cc

    weak_label = np.zeros_like(arr, dtype=np.uint8)
    # 1. erode the label to candidates region
    erode_iter = boundary_margin+line_width
    while True:
        arr_eroded = ndimage.binary_erosion(arr, iterations=erode_iter).astype(arr.dtype)
        if arr_eroded.sum() > 0:
            break
        erode_iter = erode_iter // 2
        if erode_iter == 0:
            arr_eroded = arr.copy()
            break

    # 2. random select a point
    xs, ys = np.nonzero(arr_eroded == 1)
    _index = random.choice(list(range(len(xs))))
    center = [xs[_index], ys[_index]]

    # 3. generate a region with edge length
    radius = line_width // 2
    weak_label[center[0]-radius:center[0]-radius+line_width, center[1]-radius:center[1]-radius+line_width] = 1
    weak_label[arr == 0] = 0
    return weak_label

def generate_weak_label_slice(slice_gt, line_width, boundary_margin, fg_type='scribble', bg_type='', with_box=True):
    weak_label = np.ones_like(slice_gt, dtype=np.uint8) * 255   # unlabeled

    if to_scribble:  # only bg scribble 0, generated by dilation to simulate gt
        weak_label_bg = np.zeros_like(slice_gt, dtype=np.uint8)
        # a real-like scribble
        dists = list(range(Params['loose_box_dist'][0], Params['loose_box_dist'][1]))  # hyperparams for bg scribble distance to tight box
        _iter = random.choice(dists)
        gt_dilate1 = ndimage.binary_dilation(slice_gt[0], iterations=_iter)
        gt_dilate2 = ndimage.binary_dilation(gt_dilate1, iterations=line_width)
        weak_label_bg[0][gt_dilate2 == 1] = 1
        weak_label_bg[0][gt_dilate1 == 1] = 0
        weak_label[weak_label_bg == 1] = 0
    elif to_tightbox:   # only generate tight box
        nonzero = np.nonzero(slice_gt)
        x0, x1 = nonzero[1].min(), nonzero[1].max()
        y0, y1 = nonzero[2].min(), nonzero[2].max()
        weak_label_bg = np.zeros_like(slice_gt, dtype=np.uint8)
        weak_label_bg[0, x0-line_width:x1 + 1 + line_width, y0-line_width:y1 + 1 + line_width] = 1
        weak_label_bg[0, x0:x1 + 1, y0:y1 + 1] = 0
        weak_label[weak_label_bg == 1] = 0
    elif to_point:      # only one bg point with side length 3 pixel
        weak_label_bg = np.zeros_like(slice_gt, dtype=np.uint8)
        # a scribble
        dists = list(range(Params['loose_box_dist'][0], Params['loose_box_dist'][1]))  # hyperparams for bg scribble distance to tight box
        _iter = random.choice(dists)
        gt_dilate1 = ndimage.binary_dilation(slice_gt[0], iterations=_iter)
        gt_dilate2 = ndimage.binary_dilation(gt_dilate1, iterations=1)
        weak_label_bg[0][gt_dilate2 == 1] = 1
        weak_label_bg[0][gt_dilate1 == 1] = 0
        # a point on the scribble
        nonzero = np.nonzero(weak_label_bg[0])
        indexs = len(nonzero[0])
        _index = random.choice(list(range(indexs)))
        center = (nonzero[0][_index], nonzero[1][_index])
        _side = 3; _pside = _side // 2
        weak_label_bg[weak_label_bg != 0] = 0
        weak_label_bg[0, center[0]-_pside: center[0]-_pside+_side, center[1]-_pside: center[1]-_pside+_side] = 1
        weak_label[weak_label_bg == 1] = 0

    elif with_box:
        nonzero = np.nonzero(slice_gt)
        x0, x1 = nonzero[1].min(), nonzero[1].max()
        y0, y1 = nonzero[2].min(), nonzero[2].max()
        weak_label_bg = np.zeros_like(slice_gt, dtype=np.uint8)
        weak_label_bg[0, x0:x1 + 1, y0:y1 + 1] = 255  # unlabeled
        weak_label[weak_label_bg == 0] = 0

        if to_kernelcut:    # only bg scribble 0, no other bg info
            # one scribble for BG region
            weak_label = np.ones_like(slice_gt, dtype=np.uint8) * 255  # unlabeled
            weak_label_bg = np.zeros_like(slice_gt, dtype=np.uint8)

            # a box-like scribble
            shifts = [0, 0, 0, 0]   # xleft, xright, yleft, yright
            dists = list(range(Params['loose_box_dist'][0], Params['loose_box_dist'][1]))     #hyperparams for bg scribble distance to tight box
            for ith in range(len(shifts)):
                shifts[ith] = random.choice(dists)
            weak_label_bg[0, x0-shifts[0]-line_width:x1+1+shifts[1]+line_width, y0-shifts[2]-line_width:y1+1+shifts[3]+line_width] = 1
            weak_label_bg[0, x0-shifts[0]:x1+1+shifts[1], y0-shifts[2]:y1+1+shifts[3]] = 0
            weak_label[weak_label_bg == 1] = 0
        elif bg_type == 'loose':
            weak_label = np.ones_like(slice_gt, dtype=np.uint8) * 255  # unlabeled
            shifts = [0, 0, 0, 0]  # xleft, xright, yleft, yright
            dists = list(range(Params['loose_box_dist'][0], Params['loose_box_dist'][1]))  # hyperparams for bg scribble distance to tight box
            for ith in range(len(shifts)):
                shifts[ith] = random.choice(dists)
            weak_label_bg[0, x0-shifts[0]:x1+1+shifts[1], y0-shifts[2]:y1+1+shifts[3]] = 255  # unlabeled
            weak_label[weak_label_bg == 0] = 0

    if to_tightbox:
        pass
    elif to_point or fg_type == 'point':
        weak_label_fg = gen_point_label(slice_gt[0], line_width, boundary_margin)
        weak_label_fg = weak_label_fg[np.newaxis, :, :]
        weak_label[weak_label_fg == 1] = 1
    elif fg_type == 'major_axis':
        weak_label_fg = gen_long_scribble(slice_gt[0], line_width, boundary_margin)
        weak_label_fg = weak_label_fg[np.newaxis, :, :]
        weak_label[weak_label_fg == 1] = 1
    elif fg_type == 'scribble':
        weak_label_fg = gen_two_lines(slice_gt[0], line_width, boundary_margin)
        weak_label_fg = weak_label_fg[np.newaxis, :, :]
        weak_label[weak_label_fg == 1] = 1


    return weak_label


def generate_weak_label_volume(label, n_percent=1, sample_style='uniform',
                               line_width=3, boundary_margin=5):
    weak_label = np.ones_like(label, dtype=np.uint8) * 255
    slice_ids = []
    nonzero = np.nonzero(label)
    z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
    n_slices = z1 - z0 + 1

    # fixed containing relation setting of r1, r3, r5
    all_percents = Params['all_percents']   # default [0.5, 0.3, 0.1]
    all_nums = [math.ceil(max(n_slices * _percent - 2, 0)) for _percent in all_percents]
    all_slice_ids = [[] for _ in all_percents]
    if sample_style == 'uniform':
        raise NotImplementedError
    elif sample_style == 'random':
        default_ids = [z0, z1]
        last_candidates = list(range(z0+1, z1))
        for ith, _num in enumerate(all_nums):
            if _num == 0:
                all_slice_ids[ith] = []
            else:
                all_slice_ids[ith] = random.sample(last_candidates, _num)
            last_candidates = all_slice_ids[ith]
        for ith in range(len(all_nums)):
            all_slice_ids[ith] = sorted(all_slice_ids[ith] + default_ids)
    # current slice_ids
    slice_ids = all_slice_ids[all_percents.index(n_percent)]

    for i in slice_ids:
        slice_gt = copy.deepcopy(label[i:i+1])
        weak_label_i = generate_weak_label_slice(slice_gt, line_width, boundary_margin, 
            fg_type=Params['fg_type'], bg_type=Params['bg_type'], with_box=True)
        weak_label[i:i+1] = weak_label_i

    return weak_label, slice_ids, n_slices

def select_weak_label_volume(label, n_percent=1):
    '''
    :param label:  weak labels, all slices are weak labeled
    '''
    weak_label = np.ones_like(label, dtype=np.uint8) * 255
    if to_tightbox:
        nonzero = np.nonzero(label == 0)
    else:
        nonzero = np.nonzero(label == 1)
    z0, z1 = nonzero[0].min(), nonzero[0].max()  # z axis=0
    n_slices = z1 - z0 + 1

    # 1. fixed containing relation setting of r1, r3, r5
    all_percents = Params['all_percents']  # default [0.5, 0.3, 0.1]
    all_nums = [math.ceil(max(n_slices * _percent - 2, 0)) for _percent in all_percents]
    all_slice_ids = [[] for _ in all_percents]
    if sample_style == 'uniform':
        raise NotImplementedError
    elif sample_style == 'random':
        default_ids = [z0, z1]
        last_candidates = list(range(z0 + 1, z1))
        for ith, _num in enumerate(all_nums):
            if _num == 0:
                all_slice_ids[ith] = []
            else:
                all_slice_ids[ith] = random.sample(last_candidates, _num)
            last_candidates = all_slice_ids[ith]
        for ith in range(len(all_nums)):
            all_slice_ids[ith] = sorted(all_slice_ids[ith] + default_ids)

    # 2. current slice_ids
    if n_percent in all_percents:
        slice_ids = all_slice_ids[all_percents.index(n_percent)]
    else:
        # find two nearest labels and select
        upper, lower = 0.0, 0.0
        for ith in range(len(all_percents) - 1):
            upper, lower = all_percents[ith], all_percents[ith+1]
            if upper > n_percent and lower < n_percent:
                break
        slice_ids = all_slice_ids[ith+1]
        _num = math.ceil(max(n_slices * n_percent - 2, 0))
        _num = _num - len(all_slice_ids[ith+1])
        candidates = [ele for ele in all_slice_ids[ith] if ele not in all_slice_ids[ith+1]]
        chosens = random.sample(candidates, _num)
        [slice_ids.append(ele) for ele in chosens]


    # 3. select weak labels
    for i in slice_ids:
        weak_label[i:i+1] = copy.deepcopy(label[i:i+1])

    return weak_label, slice_ids, n_slices


def load_mhd(root, phase, out_path, weak,
             n_percent=1, sample_style='uniform',
             line_width=3, boundary_margin=5):
    step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    label_log_path = os.path.join(out_path, 'label_log.json')

    paths = get_input_paths(root, phase)
    data = {}
    label_log = {}; total_slices = 0; labeled_slices = 0
    for (image_p, label_p) in tqdm(sorted(paths)):
        case_name = image_p.split('/')[-1][:6]
        image = sitk.ReadImage(image_p)
        spacing = image.GetSpacing()
        spacing = (spacing[2], spacing[0], spacing[1])
        image = io.imread(image_p, plugin='simpleitk') # z, 512, 512
        label = io.imread(label_p, plugin='simpleitk')
        # image = nib.load(image_p).get_data()  # (W, H, S)
        # label = nib.load(label_p).get_data()  # (W, H, S)

        if weak:
            weak_label, slice_ids, n_slices = generate_weak_label_volume(label, n_percent=n_percent,
                     sample_style=sample_style, line_width=line_width, boundary_margin=boundary_margin)
            label_log[case_name] = {
                'n_slices': n_slices, 'slice_ids': slice_ids, 'slice_num': len(slice_ids),
            }
            total_slices += n_slices
            labeled_slices += len(slice_ids)
        else:
            weak_label, slice_ids, n_slices = np.zeros_like(label, dtype=np.uint8), [], 0
        data[case_name] = {
            'image': image,
            'label': label,
            'weak_label': weak_label,
            'slice_ids': slice_ids,
            'num_slices': n_slices,
            'spacing': spacing
        }

        print('load  ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('load  ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('load  ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        print('slice_ids', slice_ids, n_slices)
        print(case_name, spacing)
        print(image_p)
        print(label_p)

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)

    # save label_log
    if weak:
        label_log['info_n_percent'] = n_percent
        label_log['info_sample_style'] = sample_style
        label_log['info_total_slices'] = total_slices
        label_log['info_labeled_slices'] = labeled_slices
        with open(label_log_path, 'w') as f:
            json.dump(label_log, f, cls=NpEncoder)

def load_la(root, phase, out_path, weak, n_percent=1, sample_style='uniform', line_width=3, boundary_margin=5,
            depth_transpose=True):
    step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    label_log_path = os.path.join(out_path, 'label_log.json')

    paths = get_la_paths(root, phase)
    data = {}
    label_log = {}; total_slices = 0; labeled_slices = 0
    for ith, (image_p, label_p) in enumerate(tqdm(sorted(paths))):
        case_name = 'Case%02d' % ith
        image, img_header = nrrd.read(image_p)
        label, gt_header = nrrd.read(label_p)
        label = (label == 255).astype(np.uint8)
        spacing = [1, 1, 1]     # default (1,1,1)

        if depth_transpose:
            image = np.transpose(image, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            spacing = (spacing[2], spacing[0], spacing[1])

        if weak:
            weak_label, slice_ids, n_slices = generate_weak_label_volume(label, n_percent=n_percent,
                                                                         sample_style=sample_style,
                                                                         line_width=line_width,
                                                                         boundary_margin=boundary_margin)
            label_log[case_name] = {
                'n_slices': n_slices, 'slice_ids': slice_ids, 'slice_num': len(slice_ids),
            }
            total_slices += n_slices
            labeled_slices += len(slice_ids)
        else:
            weak_label, slice_ids, n_slices = np.zeros_like(label, dtype=np.uint8), [], 0
        data[case_name] = {
            'image': image,
            'label': label,
            'weak_label': weak_label,
            'slice_ids': slice_ids,
            'num_slices': n_slices,
            'spacing': spacing
        }

        print('load  ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('load  ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('load  ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        print('slice_ids', slice_ids, n_slices)
        print(case_name, spacing)
        print(image_p)
        print(label_p)

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)

    # save label_log
    if weak:
        label_log['info_n_percent'] = n_percent
        label_log['info_sample_style'] = sample_style
        label_log['info_total_slices'] = total_slices
        label_log['info_labeled_slices'] = labeled_slices
        with open(label_log_path, 'w') as f:
            json.dump(label_log, f, cls=NpEncoder)

def load_saved(saved_path, out_path, weak, n_percent=1, ref_dir=''):
    def transform_weak_label(weak):
        '''
        Convert {bg box + fg scribble} to {bg scribble + fg scribble}
        '''
        weak_new = copy.deepcopy(weak)
        for z in range(weak.shape[0]):
            if (weak[z] == 1).any() and (weak_label[z] == 0).any():
                weak_new[z] = np.ones_like(weak_new[z]) * 255
                tmp_nonzero = np.nonzero(weak[z] != 0)
                x0, x1, y0, y1 = tmp_nonzero[0].min(), tmp_nonzero[0].max(), tmp_nonzero[1].min(), tmp_nonzero[1].max()
                weak_new[z, x0-line_width:x1+1+line_width, y0-line_width:y1+1+line_width] = 0
                weak_new[z, x0:x1 + 1, y0:y1 + 1] = 255
                weak_new[z][weak[z] == 1] = 1
        return weak_new

    # 1. load saved volumes
    step = 'load'
    prev_step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    if to_kernelcut and ref_dir!='':
        data = load_data(prev_step, other_load_path=os.path.join(os.path.dirname(saved_path), ref_dir))
    else:
        data = load_data(prev_step, other_load_path=saved_path)
    label_log_path = os.path.join(out_path, 'label_log.json')
    save_data = {}; label_log = {}
    total_slices = 0; labeled_slices = 0

    # 2. select weak labels
    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']
        spacing = data[case_name]['spacing'][0]

        if weak:
            if to_kernelcut and ref_dir!='':
                weak_label = transform_weak_label(weak_label)
                slice_ids, n_slices = [], -1
            else:
                weak_label, slice_ids, n_slices = select_weak_label_volume(weak_label, n_percent=n_percent)

            label_log[case_name] = {
                'n_slices': n_slices, 'slice_ids': slice_ids, 'slice_num': len(slice_ids),
            }
            total_slices += n_slices; labeled_slices += len(slice_ids)
        save_data[case_name] = {
            'image': image,
            'label': label,
            'weak_label': weak_label,
            'slice_ids': slice_ids,
            'num_slices': n_slices,
            'spacing': spacing
        }

        print('load  ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('load  ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('load  ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        print('slice_ids', slice_ids, n_slices)
        print(case_name, spacing)

        save_case(step, case_name, image, label, weak_label, save_data, save_nii_dir, save_mat_dir)

    # 3. save label_log
    if weak:
        label_log['info_n_percent'] = n_percent
        label_log['info_sample_style'] = sample_style
        label_log['info_total_slices'] = total_slices
        label_log['info_labeled_slices'] = labeled_slices
        with open(label_log_path, 'w') as f:
            json.dump(label_log, f, cls=NpEncoder)


def load_nii(root, phase, out_path, weak,
             n_percent=1, sample_style='uniform',
             line_width=3, boundary_margin=5,
             organ_label=None, depth_transpose=False,
             ):
    step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    label_log_path = os.path.join(out_path, 'label_log.json')

    paths = get_nii_paths(root, phase)
    data = {}
    label_log = {}; total_slices=0; labeled_slices=0

    for (image_p, label_p) in tqdm(sorted(paths)):
        case_id = image_p.split('/')[-1][8:10]
        case_name = 'Case%02d' % (int(case_id))
        nii_image = sitk.ReadImage(label_p)
        spacing = nii_image.GetSpacing()
        image = nib.load(image_p).get_data()
        label = nib.load(label_p).get_data()  # (W, H, S)
        if organ_label is not None:
            label = (label == organ_label).astype(np.uint8)
        if depth_transpose:
            image = np.transpose(image, (2, 0, 1))
            label = np.transpose(label, (2, 0, 1))
            spacing = (spacing[2], spacing[0], spacing[1])

        if weak:
            weak_label, slice_ids, n_slices = generate_weak_label_volume(label, n_percent=n_percent,
                    sample_style=sample_style, line_width=line_width, boundary_margin=boundary_margin)
            label_log[case_name] = {
                'n_slices': n_slices, 'slice_ids': slice_ids, 'slice_num': len(slice_ids),
            }
            total_slices += n_slices
            labeled_slices += len(slice_ids)
        else:
            weak_label, slice_ids, n_slices = np.zeros_like(label, dtype=np.uint8), [], 0

        data[case_name] = {
            'image': image,
            'label': label,
            'weak_label': weak_label,
            'slice_ids': slice_ids,
            'num_slices': n_slices,
            'spacing': spacing
        }

        print('load  ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('load  ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('load  ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        print('slice_ids', slice_ids, n_slices)
        print(case_name, spacing)
        print(image_p)
        print(label_p)

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)

    # save label_log
    if weak:
        label_log['info_n_percent'] = n_percent
        label_log['info_sample_style'] = sample_style
        label_log['info_total_slices'] = total_slices
        label_log['info_labeled_slices'] = labeled_slices
        with open(label_log_path, 'w') as f:
            json.dump(label_log, f, cls=NpEncoder)

def resample(out_path):
    step = 'resample'
    prev_step = 'load'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']
        spacing = data[case_name]['spacing'][0]
        
        print('before', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('before', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('before', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        if max(spacing) / min(spacing) > 3:
            anisotropic = True
        else:
            anisotropic = False
        if re_sample[0] != 0 or re_sample[1] != 0 or re_sample[2] != 0 or (re_sample == spacing):
            zoom_factor = [spacing[i] / re_sample[i] for i in range(3)]
            if anisotropic:
                orders = [3, 1, 0, 0]
            else:
                orders = [3, 1, 3, 1]
            
            assert np.argmax(spacing) == 0  # z axis
            image = ndimage.zoom(image, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[0])
            label = ndimage.zoom(label, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
            weak_label_fg = (weak_label == 1).astype(np.uint8)
            weak_label_fg = ndimage.zoom(weak_label_fg, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
            if to_kernelcut or to_tightbox or to_scribble:
                weak_label_bg = (weak_label == 0).astype(np.uint8)
                weak_label_bg = ndimage.zoom(weak_label_bg, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
                weak_label = np.ones_like(label, dtype=np.uint8) * 255
                weak_label[weak_label_fg == 1] = 1
                weak_label[weak_label_bg == 1] = 0
            else:
                weak_label_bg = (weak_label != 0).astype(np.uint8)
                weak_label_bg = ndimage.zoom(weak_label_bg, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
                weak_label = np.ones_like(label, dtype=np.uint8) * 255
                weak_label[weak_label_fg == 1] = 1
                weak_label[weak_label_bg == 0] = 0

            image = ndimage.zoom(image, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[2])
            label = ndimage.zoom(label, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label_fg = (weak_label == 1).astype(np.uint8)
            weak_label_fg = ndimage.zoom(weak_label_fg, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label_bg = (weak_label != 0).astype(np.uint8)
            weak_label_bg = ndimage.zoom(weak_label_bg, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label = np.ones_like(label, dtype=np.uint8) * 255
            weak_label[weak_label_fg == 1] = 1
            weak_label[weak_label_bg == 0] = 0

        data[case_name]['image'] = image
        data[case_name]['label'] = label
        data[case_name]['weak_label'] = weak_label

        print('after ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum() / image.shape[0] / image.shape[1] / image.shape[2])
        print('after ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum() / label.shape[0] / label.shape[1] / label.shape[2])
        print('after ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        print(case_name, spacing)

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)


def align_volume_center(out_path):
    step = 'align'
    prev_step = 'resample'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    volume_shape = np.array([data[case_name]['image'].shape for case_name in data.keys()])
    max_shape = np.max(volume_shape, axis=0)
    min_shape = np.min(volume_shape, axis=0)
    max_center = max_shape // 2
    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        image_shape = np.array(image.shape)
        image_center = image_shape // 2
        start = (max_center - image_center).astype(int)
        end = (start + image_shape).astype(int)

        print(case_name)
        print('before', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum())
        print('before', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum())
        print('before', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        image_new = np.zeros(max_shape).astype(np.int16)
        image_new[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = copy.deepcopy(image)
        label_new = np.zeros(max_shape).astype(np.int8)
        label_new[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = copy.deepcopy(label)
        weak_label_new = np.ones(max_shape).astype(np.uint8) * 255
        for z in range(weak_label.shape[0]):
            if ((weak_label[z] == 1).any() and (weak_label[z] == 0).any() and not to_tightbox) \
                    or ((weak_label[z] == 0).any() and to_tightbox):
                if to_kernelcut or to_scribble or to_tightbox:
                    weak_label_new[start[0] + z] = 255
                else:
                    weak_label_new[start[0] + z] = 0
                # modify those all 0 slices to all 255 (which generated by resample, located in extreme slice)
                if not (weak_label[z] != 0).any():
                    continue
                weak_label_new[start[0] + z, start[1]:end[1], start[2]:end[2]] = copy.deepcopy(weak_label[z])

        image, label, weak_label = image_new, label_new, weak_label_new
        data[case_name]['image'] = image
        data[case_name]['label'] = label
        data[case_name]['weak_label'] = weak_label

        print('after ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum())
        print('after ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum())
        print('after ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)


def compute_roi_from_label(data, weak):
    roi_priors = []
    for case_name in data.keys():
        label = data[case_name]['label']
        if weak:
            weak_label = data[case_name]['weak_label']
            label = np.zeros_like(weak_label, dtype=np.uint8)
            for z in range(weak_label.shape[0]):
                if ((weak_label[z] == 1).any() and (weak_label[z] == 0).any()) \
                        or (to_tightbox and (weak_label[z] == 0).any()):
                    # calculate roi from bg scribble or box
                    if to_kernelcut or to_scribble or to_tightbox:
                        tmp_nonzero = np.nonzero(weak_label[z] == 0)
                        x0, x1, y0, y1 = tmp_nonzero[0].min(), tmp_nonzero[0].max(), tmp_nonzero[1].min(), tmp_nonzero[1].max()
                        x0, x1, y0, y1 = x0 + line_width, x1 - line_width, y0 + line_width, y1 - line_width
                        label[z, x0:x1+1, y0:y1+1] = 1
                    else:
                        # if bg all 0, roi = those nonzero regions
                        label[z] = (weak_label[z] != 0).astype(np.uint8)
        nonzero = np.nonzero(label)
        a0, a1, b0, b1, c0, c1 = nonzero[0].min(), nonzero[0].max() + 1, \
                                 nonzero[1].min(), nonzero[1].max() + 1, \
                                 nonzero[2].min(), nonzero[2].max() + 1
        roi_priors.append([a0, a1, b0, b1, c0, c1])
        print(case_name, [a0, a1, b0, b1, c0, c1])

    roi_priors = np.array(roi_priors)
    roi_max = np.max(roi_priors, axis=0)
    roi_min = np.min(roi_priors, axis=0)
    roi = np.array([roi_min[0], roi_max[1],
                    roi_min[2], roi_max[3],
                    roi_min[4], roi_max[5]])

    print('shape', label.shape)
    print('a0, a1, b0, b1, c0, c1')
    print(roi_priors)
    print('roi', roi, [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]])
    return roi


def enlarge_roi(roi, margin_ratio):
    if margin_ratio > 0:
        shape = [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
        padding = [0, int(shape[1]*margin_ratio), int(shape[2]*margin_ratio)]     # no extension in z axis
        roi[0] -= padding[0] // 2; roi[1] += padding[0] - padding[0] // 2
        roi[2] -= padding[1] // 2; roi[3] += padding[1] - padding[1] // 2
        roi[4] -= padding[2] // 2; roi[5] += padding[2] - padding[2] // 2
    print('roi after enlarge', roi, [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]])
    return roi


def down_size_roi(roi, ratio_all):
    for r_i in range(3):
        ratio = ratio_all[r_i]
        if ratio < 1:
            for i in range(2):
                roi[i + r_i * 2] = round(roi[i + r_i * 2] * ratio)
            print('roi after down size', roi, 
                [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]])
    return roi


def compute_round_2m(size):
    down_2m = int(math.log(size) / math.log(2)) - 2    # feature size >= 4 -> 2**2
    down_n = int(math.pow(2, down_2m))
    return down_2m, down_n


def fit_roi_to_network_down_n(roi, axis, down_n=None):
    size = roi[axis*2+1] - roi[axis*2]
    if down_n is None:
        _, down_n = compute_round_2m(size)
    if size % down_n != 0:   # a least pad something (1-down_n)? or (0-down_n-1)
        padding = down_n - size % down_n
        size += padding
        roi[axis*2] -= padding // 2; roi[axis*2+1] += padding - padding // 2
        if axis == 0:
            _, down_n = compute_round_2m(size)    # e.g. size 63->64, down_2m 3->4
    print('roi, down_n, axis', roi, down_n, axis,
        [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]])
    return roi, down_n


def compute_down_ratio_to_memory(roi, memory, changed_axis=2):
    ratio_all = [1, 1, 1]
    down_n_all = [1, 1, 1]
    shape = [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
    # shuailin extension
    down_scale = float(memory) / shape[0] / shape[1] / shape[2]

    if down_scale >= 1:
        shape_top = shape
    elif changed_axis == 2:   # XY plane downsample
        r0 = math.sqrt(down_scale)
        shape_top = [shape[0], int(r0 * shape[1]), int(r0 * shape[2])]
    elif changed_axis == 1:     # Z plane downsample
        r0 = down_scale
        shape_top = [int(r0 * shape[0]), shape[1], shape[2]]
    elif changed_axis == 3:     # Z,XY downsample
        r0 = math.sqrt(down_scale); r1 = math.sqrt(r0)
        shape_top = [int(r0 * shape[0]), int(r1 * shape[1]), int(r1 * shape[2])]

    start_axis = 1 if changed_axis==2 else 0
    for i in range(start_axis, 3):
        size = shape_top[i]
        down_2m, down_n = compute_round_2m(size)
        inv_padding = size % down_n
        size -= inv_padding
        ratio_all[i] = size / shape[i]
        down_n_all[i] = down_n
        assert down_2m == (int(math.log(size) / math.log(2)) - 2)
    ratio_xy = min(ratio_all)
    if ratio_xy >= 1:
        down_n_all = [down_n_all[0], 1, 1]
    else:
        ratio_all = [ratio_all[0], ratio_xy, ratio_xy]
        down_n_all[1], down_n_all[2] = min(down_n_all[1:]), min(down_n_all[1:])

    print('ratio_all, down_n_all', ratio_all, down_n_all)
    return ratio_all, down_n_all


def check_roi(roi, down_n_all):
    shape = [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
    assert down_n_all[1] == down_n_all[2]
    for i in range(3):
        assert shape[i] % down_n_all[i] == 0
        assert shape[i] // down_n_all[i] >= 4
    assert shape[1] // down_n_all[1] < 8 or shape[2] // down_n_all[2] < 8
    print('roi, down_n_all, shape', roi, down_n_all,
        [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]])


def get_min_down_n(roi):
    down_n_all = []
    for axis in [1, 2]:
        size = roi[axis*2+1] - roi[axis*2]
        _, down_n = compute_round_2m(size)
        if size % down_n != 0:   # a least pad something (1-down_n)? or (0-down_n-1)
            padding = down_n - size % down_n
            size += padding
            _, down_n = compute_round_2m(size)    # e.g. size 63->64, down_2m 3->4
        down_n_all.append(down_n)
    return min(down_n_all)


def compute_down_ratio_and_roi(data, margin_ratio, memory, weak, changed_axis=2):
    roi = compute_roi_from_label(data, weak)
    roi = enlarge_roi(roi, margin_ratio)
    if changed_axis == 2:
        roi, down_n_z = fit_roi_to_network_down_n(roi, axis=0)
        ratio_all, down_n_all = compute_down_ratio_to_memory(roi, memory, changed_axis=changed_axis)
        down_n_all[0] = down_n_z
        roi = down_size_roi(roi, ratio_all)
    else:
        ratio_all, down_n_all = compute_down_ratio_to_memory(roi, memory, changed_axis=changed_axis)
        roi = down_size_roi(roi, ratio_all)

    down_n_min = get_min_down_n(roi)
    roi, down_n_all[1] = fit_roi_to_network_down_n(roi, axis=1, down_n=down_n_min)
    roi, down_n_all[2] = fit_roi_to_network_down_n(roi, axis=2, down_n=down_n_min)

    check_roi(roi, down_n_all)
    return ratio_all, roi, down_n_all


def crop_roi(out_path, margin_ratio, memory, weak, params_only=False, changed_axis=2):
    step = 'crop'
    prev_step = 'align'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    zoom_factor, roi_orig, down_n_all = compute_down_ratio_and_roi(data, margin_ratio, memory, weak, changed_axis=changed_axis)
    # ipdb.set_trace()
    if params_only:
        return zoom_factor, roi, down_n_all
    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        print(case_name)
        print('before    ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum())
        print('before    ', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum())
        print('before    ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        orders = [3, 1, 3, 1]
        image = ndimage.zoom(image, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[0])
        label = ndimage.zoom(label, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
        weak_label_fg = (weak_label == 1).astype(np.uint8)
        weak_label_fg = ndimage.zoom(weak_label_fg, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
        weak_label_bg = (weak_label != 0).astype(np.uint8)
        weak_label_bg = ndimage.zoom(weak_label_bg, zoom=[1.0, zoom_factor[1], zoom_factor[2]], order=orders[1])
        weak_label = np.ones_like(label, dtype=np.uint8) * 255
        weak_label[weak_label_fg == 1] = 1
        weak_label[weak_label_bg == 0] = 0

        # downsample along Z axis
        if changed_axis != 2:
            image = ndimage.zoom(image, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[2])
            label = ndimage.zoom(label, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label_fg = (weak_label == 1).astype(np.uint8)
            weak_label_fg = ndimage.zoom(weak_label_fg, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label_bg = (weak_label != 0).astype(np.uint8)
            weak_label_bg = ndimage.zoom(weak_label_bg, zoom=[zoom_factor[0], 1.0, 1.0], order=orders[3])
            weak_label = np.ones_like(label, dtype=np.uint8) * 255
            weak_label[weak_label_fg == 1] = 1
            weak_label[weak_label_bg == 0] = 0

        print('after down', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum())
        print('after down', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum())
        print('after down', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        roi = roi_orig.copy()
        # extend shape if roi exceed image shape (current only consider z axis)
        if roi[0] < 0 or roi[1] > image.shape[0]:
            new_shape = (roi[1] - roi[0], image.shape[1], image.shape[2])
            image_new = np.zeros(new_shape).astype(np.int16)
            label_new = np.zeros(new_shape).astype(np.int8)
            weak_label_new = np.ones(new_shape).astype(np.uint8) * 255
            if roi[0] < 0:
                start = 0-roi[0]; end = image.shape[0] - roi[0]
                roi[1] = roi[1] - roi[0]; roi[0] = 0
            elif roi[1] > image.shape[0]:
                start = roi[0]; end = image.shape[0] + roi[0]
            image_new[start:end, :, :] = image.copy()
            label_new[start:end, :, :] = label.copy()
            weak_label_new[start:end, :, :] = weak_label.copy()
            image, label, weak_label = image_new, label_new, weak_label_new
            print('modified before shape', image.shape)
            print('modified roi', roi)
        image = image[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        label = label[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        weak_label = weak_label[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        data[case_name]['image'] = image
        data[case_name]['label'] = label
        data[case_name]['weak_label'] = weak_label

        print('after crop', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum())
        print('after crop', label.shape, type(label[0, 0, 0]), label.max(),
              label.sum())
        print('after crop', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)
    return zoom_factor, roi, down_n_all


def normalize_intensity(out_path, data_name='Promise'):
    step = 'normalize'
    prev_step = 'crop'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        print(case_name)
        print('before', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum(), image.min())

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


        print('mean std', mean, std, image.shape,
            image.shape[0]*image.shape[1]*image.shape[2])

        print('after ', image.shape, type(image[0, 0, 0]), image.max(),
              image.sum(), image.min())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)

def expand_bg(out_path):
    step = 'expand'
    prev_step = 'normalize'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        print(case_name)
        print('before', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())
        
        nonzero = np.nonzero(label)
        z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
        weak_label[:z0] = 0
        weak_label[z1+1:] = 0
        data[case_name]['weak_label'] = weak_label

        print('after ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)

def modify_bg(out_path):
    step = 'expand_bg'
    prev_step = 'expand'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    data = load_data(prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        print(case_name)
        print('before', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        for z in range(weak_label.shape[0]):
            # out of tight box: set as 0
            if (weak_label[z] == 0).any() and (weak_label[z] == 255).any():
                nonzero = np.nonzero(weak_label[z] == 0)
                x1, x2, y1, y2 = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
                label_new = np.zeros_like(weak_label[z])
                label_new[x1: x2, y1: y2] = 255
                label_new[weak_label[z] == 0] = 0
                weak_label[z] = label_new

        data[case_name]['weak_label'] = weak_label

        print('after ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)


def graphcut_label(out_path):
    def graphcut_3d(image, weak_label):
        import imcut.pycut as pspc
        '''
        need input 1(FG)
        '''

        seeds = weak_label.copy()
        seeds[seeds == 0] = 2  # background
        seeds[seeds == 255] = 0  # unlabel

        # process to set fg
        start = -1; end = -1; end2 = -1
        for z in range(weak_label.shape[0]):
            if (weak_label[z] == 255).any() and (weak_label[z] == 0).any():
                if start == -1:
                    start = z
                # center 1
                nonzero = np.nonzero(weak_label[z] == 255)
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
                center = ((x0 + x1) //2, (y0 + y1) //2)
                seeds[z, center[0], center[1]] = 1
                end2 = end; end = z
        seeds[start+(end-start)//2, :, :][seeds[start+(end-start)//2, :, :] == 1] = 0

        # run
        igc = pspc.ImageGraphCut(image, voxelsize=[1, 1, 1])
        igc.set_seeds(seeds)
        igc.run()
        # save results
        pred = igc.segmentation
        pred = (1 - pred).astype(np.uint8)
        # weak_label[np.logical_and(weak_label == 255, pred == 1)] = 1

        # post process to filter unlabeled slice noise
        pred2 = weak_label.copy()
        for z in range(weak_label.shape[0]):
            if (weak_label[z] == 255).any() and (weak_label[z] == 0).any():
                pred2[z][weak_label[z] == 255] = pred[z][weak_label[z] == 255]
        pred = pred2

        '''
        candiate = np.zeros_like(pred)
        for ith in [end, end2]:
            candiate[ith][weak_label[ith] == 255] = 1
        cc_labels = skimage.measure.label(pred)
        mode = scipy.stats.mode(cc_labels[candiate==1])[0]
        if mode == 0:
            mode = 1
        pred = (cc_labels == mode)
        '''
        return pred

    def grabcut_2ds(image, weak_label):

        '''
        grabcut on each labeled slice, stack them
        '''
        import cv2
        pred = weak_label.copy()
        for z in range(weak_label.shape[0]):
            if (weak_label[z] == 255).any() and (weak_label[z] == 0).any():
                # image mode
                _slice = image[z]  #:z+1
                _slice = (1.0 * (_slice - _slice.min()) / (_slice.max() - _slice.min()) * 255).astype(np.uint8)
                _slice = np.expand_dims(_slice, axis=2)
                _slice = np.repeat(_slice, 3, axis=2)
                _slice = np.ascontiguousarray(_slice)

                # bounding box
                mask = np.zeros(_slice.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                nonzero = np.nonzero(weak_label[z] == 255)
                rect = (nonzero[1].min(), nonzero[0].min(), nonzero[1].max() - nonzero[1].min(),
                        nonzero[0].max() - nonzero[0].min())

                '''
                # check rect
                fig, ax = plt.subplots()
                ax.imshow(_slice)
                rect_patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect_patch)
                plt.show()
                save_fig_path = save_dir + '%02d_rect.png' % (z)
                plt.savefig(save_fig_path)
                plt.close()
                '''

                # grabcut
                try:
                    cv2.grabCut(_slice, mask, rect, bgdModel, fgdModel, 5,
                                cv2.GC_INIT_WITH_RECT)  # mask, bgdModel, fgdModel =
                except:
                    ipdb.set_trace()
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                pred[z] = mask2
        return pred


    step = 'graphcut'
    # prev_step = 'expand'
    prev_step = 'expand_bg'
    save_nii_dir, save_mat_dir = make_save_dirs(step, out_path)
    print('Loading data: %s' % prev_step)
    data = load_data(prev_step)
    print('Loaded data: %s' % prev_step)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        print(case_name)
        print('before', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        # pred = graphcut_3d(image, weak_label)
        pred = grabcut_2ds(image, weak_label)   # from 2021.03.12

        weak_label = pred
        data[case_name]['weak_label'] = weak_label

        print('after ', weak_label.shape, type(weak_label[0, 0, 0]), weak_label.max(),
              (weak_label == 0).sum(), (weak_label == 1).sum(), (weak_label == 255).sum())

        save_case(step, case_name, image, label, weak_label, data, save_nii_dir, save_mat_dir)


def get_train_val_split(save_split_dir, data_name='Promise'):
    if data_name in ['Promise', 'promise']:
        reference = {
            'train': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 17, 18, 20,
                      21, 22, 23, 27, 28, 29, 33, 34,
                      35, 36, 37, 38, 39, 40, 41, 42,
                      43, 44, 45, 46, 47, 49],
            'val': [2, 16, 19, 24, 25, 26, 30, 31, 32, 48]
        }

        pre_split = True

        if pre_split:
            for phase in ['train', 'val']:
                with open(os.path.join(save_split_dir, '{}.txt'.format(phase)), 'w') as f:
                    for case_id in reference[phase]:
                        case_name = 'Case' + str(case_id).zfill(2)  # '{:2d}'.format(case_id)
                        f.write(case_name + '\n')
                        # print(phase, case_name)
        else:
            case_names = {'train': [], 'val': []}
            for phase in ['train', 'val']:
                prostate_2d_dir = '/group/lishl/weak_datasets/Promise12/train_slices_s1/{}/img'.format(phase)
                for tmp_path, dirs, files in os.walk(prostate_2d_dir):
                    if len(dirs) == 0:
                        files = sorted(files)
                        for file in files:
                            prefix = file[:6]
                            prefix = 'C' + prefix[1:]
                            if prefix not in case_names[phase]:
                                case_names[phase].append(prefix)

                check = [int(x[-2:]) for x in case_names[phase]]
                # ipdb.set_trace()
                if (np.array(sorted(check)) != np.array(sorted(reference[phase]))).astype(int).sum() > 0:
                    ipdb.set_trace()

                with open(os.path.join(save_split_dir, '{}.txt'.format(phase)), 'w') as f:
                    for case_name in case_names[phase]:
                        f.write(case_name + '\n')
                        # print(phase, case_name)

    elif data_name in ['Segthor', 'segthor']:
        seed = 0
        total_list = list(range(1, 41))
        val_num = 10
        random.seed(seed)

        val_list = random.sample(total_list, val_num)
        train_list = [it for it in total_list if it not in val_list]

        with open(os.path.join(save_split_dir, 'train.txt'), 'w') as f:
            for it in train_list:
                f.write('Case%02d\n' % (it))
        with open(os.path.join(save_split_dir, 'val.txt'), 'w') as f:
            for it in val_list:
                f.write('Case%02d\n' % (it))

    elif data_name in ['LA']:
        train_name_list, val_name_list = [], []
        for _phase in ['train', 'val']:
            with open(os.path.join(root, '%s_split.list' % _phase)) as f:
                for line in f:
                    subdir = line.strip()
                    if _phase == 'train':
                        train_name_list.append(subdir)
                    elif _phase == 'val':
                        val_name_list.append(subdir)
        train_num = len(train_name_list)
        train_id_list = ['Case%02d' % ith for ith in range(train_num)]
        val_id_list = ['Case%02d' % (ith + train_num) for ith in range(len(val_name_list))]

        with open(os.path.join(save_split_dir, 'train_name_dict.txt'), 'w') as f:
            for it, name in zip(train_id_list, train_name_list):
                f.write('%s\t%s\n' % (it, name))
        with open(os.path.join(save_split_dir, 'val_name_dict.txt'), 'w') as f:
            for it, name in zip(val_id_list, val_name_list):
                f.write('%s\t%s\n' % (it, name))

        with open(os.path.join(save_split_dir, 'train.txt'), 'w') as f:
            for it in train_id_list:
                f.write('%s\n' % (it))
        with open(os.path.join(save_split_dir, 'val.txt'), 'w') as f:
            for it in val_id_list:
                f.write('%s\n' % (it))



    else:
        raise NotImplementedError
    print('Train/Val Split Done!')


def compute_final_spacing(re_sample, zoom_factor):
    final_spacing = [re_sample[i] / zoom_factor[i] for i in range(3)]
    print('final spacing', final_spacing)
    return final_spacing


def compute_num_labeled_slices(out_path):
    prev_step = 'load'
    data = load_data(prev_step)
    num_slices, num_labeled_slices = 0, 0
    for case_name in data.keys():
        num_slices += data[case_name]['num_slices'][0, 0]
        try:
            num_labeled_slices += len(data[case_name]['slice_ids'][0])
        except:
            num_labeled_slices += 0
    return num_slices, num_labeled_slices


def save_params(data, save_dir, name='params.json'):
    save_path = os.path.join(save_dir, name)
    with open(save_path, 'w') as f:
        json.dump(data, f, cls=NpEncoder)


def load_params(save_dir, name='params.json'):
    save_path = os.path.join(save_dir, name)
    with open(save_path, 'r') as f:
        json.load(f)


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm
    return res.astype(np.uint8)


def load_3d_to_2d_midlbox(out_path, Params, normalized=True, post_process='midl'):
    shape = [256, 256]
    folders = ['img', 'gt', 'box', 'thickbox']
    phases = ['train', 'val']
    subdir = '2d_midl'
    save_2d_dir = os.path.join(out_path, subdir)
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    # vis_dir = os.path.join(save_2d_dir, 'visualize')
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)
    sub_dirs = {}
    for phase in phases:
        phase_dir = os.path.join(save_2d_dir, phase)
        sub_dirs[phase] = {}
        for folder in folders:
            sub_dirs[phase][folder] = os.path.join(phase_dir, folder)
            if not os.path.exists(sub_dirs[phase][folder]):
                os.makedirs(sub_dirs[phase][folder])
    
    ori_splits_dirs = {phase: os.path.join(out_path, 'splits', phase + '.txt') for phase in ['train', 'val']}
    ori_splits = {phase: [] for phase in ['train', 'val']}
    for phase, ori_split_dir in ori_splits_dirs.items():
        with open(ori_split_dir, 'r') as f:
            ori_splits[phase] = f.readlines()
            ori_splits[phase] = [line.strip() for line in ori_splits[phase]]
    
    if Params['data_name'] == 'Segthor':
        mean_std = {'mean': 20.78, 'std': 180.50}
        clip_window = [-986, 271]
    # if Params['data_name'] == 'Promise':
    #     save_case_mean_std(out_path, normalized=True, post_process=post_process)

    prev_step = 'load'
    data = load_data(prev_step)
    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        if normalized:
            # if Params['data_name'] == 'Promise':
            #     mean_std = {'mean': image.mean(), 'std': image.std()}
            if Params['data_name'] == 'Segthor':
                image[image < clip_window[0]] = clip_window[0]
                image[image > clip_window[1]] = clip_window[1]
            image = norm_arr(image)
            # image = image.astype(np.float32)
            # image = (image - mean_std['mean']) / mean_std['std']
            # if Params['data_name'] == 'Promise':
            #     image[image < -3] = -3  # normal distribution
            #     image[image > 3] = 3
            # image = (image - image.min()) / (image.max() - image.min()) * 255.0

        nonzero = np.nonzero(label)
        z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
        weak_label[:z0] = 0
        weak_label[z1+1:] = 0

        count = 0
        if case_name in ori_splits['train']:
            phase = 'train'
        elif case_name in ori_splits['val']:
            phase = 'val'
        else:
            raise NotImplementedError

        for z in range(image.shape[0]):
            image_name = f"{case_name}_{0}_{z:04d}.png"
            # image_name = case_name + '_' + '%04d' % count
            print(image_name, image[z].min(), image[z].max(), image.shape)

            image_z = copy.deepcopy(image[z])
            label_z = copy.deepcopy(label[z])
            weak_label_z = copy.deepcopy(weak_label[z])
            if (weak_label_z == 255).all() and phase == 'train':
                print('Skip unlabeled slice for train', z)
                continue

            box_z = copy.deepcopy(label[z])
            if (box_z == 1).any():
                nonzero = np.nonzero((box_z == 1).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                box_z[x0:x1, y0:y1] = 1

            thickbox_z = copy.deepcopy(weak_label[z])
            if (thickbox_z == 255).any() and (thickbox_z == 0).any():
                nonzero = np.nonzero((thickbox_z == 255).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                thickbox_z[x0:x1, y0:y1] = 1

            resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
            image_z: np.ndarray = resize_(image_z, shape).astype(np.uint8)
            label_z: np.ndarray = resize_(label_z, shape).astype(np.uint8)
            box_z: np.ndarray = resize_(box_z, shape).astype(np.uint8)
            thickbox_z: np.ndarray = resize_(thickbox_z, shape).astype(np.uint8)
            # weak_label_z: np.ndarray = resize_(weak_label_z, shape).astype(np.uint8)
            assert 0 <= image_z.min() and image_z.max() <= 255  # The range might be smaller
            assert set(uniq(label_z)).issubset(set(uniq(label)))
            assert set(uniq(box_z)).issubset(set(uniq(label)))
            assert set(uniq(thickbox_z)).issubset(set(uniq(label))) or set(uniq(thickbox_z)).issubset(set([255]))
            # assert set(uniq(weak_label_z)).issubset(set(uniq(weak_label)))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(os.path.join(sub_dirs[phase]['img'], image_name), image_z)
                imsave(os.path.join(sub_dirs[phase]['gt'], image_name), label_z)
                imsave(os.path.join(sub_dirs[phase]['box'], image_name), box_z)
                imsave(os.path.join(sub_dirs[phase]['thickbox'], image_name), thickbox_z)

            # if 'kernelcut' in post_process and (weak_label_z == 0).any() and (weak_label_z == 255).any():
                # nonzero = np.nonzero((weak_label_z == 255).astype(int))
                # x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                # weak_label_z[:x0 - 3, :] = 255
                # weak_label_z[x1 + 3:, :] = 255
                # weak_label_z[:, :y0 - 3] = 255
                # weak_label_z[:, y1 + 3:] = 255

            # scipy.io.savemat(os.path.join(sub_dirs[phase]['image'], image_name + '.mat'), {'image': image[z]})
            # scipy.io.savemat(os.path.join(sub_dirs[phase]['label'], image_name + '.mat'), {'label': label[z]})
            # scipy.io.savemat(os.path.join(sub_dirs[phase]['weak_label'], image_name + '.mat'), {'weak_label': weak_label_z})
            # if int(case_name[-2:]) < 10:
            #     visualize_2d(image_name, image_z, label_z, thickbox_z, vis_dir, box_z)
            # if phase == 'val':
            #     split_files[phase].write(image_name + '\n')
            # else:
            #     if (weak_label[z] != 0).sum() == 0:
            #         split_files['train_bg'].write(image_name + '\n')
            #         split_files['train_fgbg'].write(image_name + '\n')
            #         split_files['train_fgbgun'].write(image_name + '\n')
            #     elif (weak_label[z] == 1).sum() > 0:
            #         split_files['train_fg'].write(image_name + '\n')
            #         split_files['train_fgbg'].write(image_name + '\n')
            #         split_files['train_fgbgun'].write(image_name + '\n')
            #     elif (weak_label[z] == 0).sum() == 0 and (weak_label[z] == 1).sum() == 0:
            #         split_files['train_fgbgun'].write(image_name + '\n')
            #     else:
            #         raise NotImplementedError
            count += 1
    # for _, f in split_files.items():
    #     f.close()
    check_data_for_midl(out_path, subdir='2d_midl') # subdir='2d_midl-Aug' or '2d_midl'
    ipdb.set_trace()


def check_data_for_midl(out_path, subdir='2d_midl'):
    folders = ['img', 'gt', 'box', 'thickbox']
    phases = ['train', 'val']
    save_2d_dir = os.path.join(out_path, subdir)
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    vis_dir = os.path.join(save_2d_dir, 'visualize')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    sub_dirs = {}
    for phase in phases:
        phase_dir = os.path.join(save_2d_dir, phase)
        sub_dirs[phase] = {}
        for folder in folders:
            sub_dirs[phase][folder] = os.path.join(phase_dir, folder)
            if not os.path.exists(sub_dirs[phase][folder]):
                os.makedirs(sub_dirs[phase][folder])
    
    count = {phase: {a: 0 for a in ['0', '1', '2', '3', '4', 'C']} for phase in phases}
    for phase in phases:
        for tmp_path, dirs, files in os.walk(sub_dirs[phase]['img']):
            if len(dirs) == 0:
                files = sorted(files)
                for file in files:
                    count[phase][file[0]] += 1
                    if 'png' in file and count[phase][file[0]] < 200 and count[phase][file[0]] > 100:
                        img = np.array(Image.open(os.path.join(sub_dirs[phase]['img'], file)))
                        gt = np.array(Image.open(os.path.join(sub_dirs[phase]['gt'], file)))
                        box = np.array(Image.open(os.path.join(sub_dirs[phase]['box'], file)))
                        thickbox = np.array(Image.open(os.path.join(sub_dirs[phase]['thickbox'], file)))
                        assert 0 <= img.min() and img.max() <= 255
                        assert set(uniq(gt)).issubset(set([0, 1]))
                        assert set(uniq(box)).issubset(set([0, 1]))
                        if phase == 'train':
                            assert set(uniq(thickbox)).issubset(set([0, 1]))
                        else:
                            assert set(uniq(thickbox)).issubset(set([0, 1])) or set(uniq(thickbox)).issubset(set([255]))
                        visualize_2d(phase+file, img, gt, thickbox, vis_dir, box)
    ipdb.set_trace()


def load_3d_to_2d_midlbox_test(out_path, Params, normalized=True, post_process='midl'):
    # out_path = '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_1.0'
    out_path = '/group/lishl/weak_datasets/0108_SegTHOR/1011_test_weak_percent_1.0'
    shape = [256, 256]
    folders = ['img', 'gt', 'box', 'thickbox']
    phases = ['train', 'val']
    subdir = '2d_midl'
    save_2d_dir = os.path.join(out_path, subdir)
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    # vis_dir = os.path.join(save_2d_dir, 'visualize')
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)
    sub_dirs = {}
    for phase in phases:
        phase_dir = os.path.join(save_2d_dir, phase)
        sub_dirs[phase] = {}
        for folder in folders:
            sub_dirs[phase][folder] = os.path.join(phase_dir, folder)
            if not os.path.exists(sub_dirs[phase][folder]):
                os.makedirs(sub_dirs[phase][folder])
    
    if Params['data_name'] == 'Segthor':
        mean_std = {'mean': 20.78, 'std': 180.50}
        clip_window = [-986, 271]
    # if Params['data_name'] == 'Promise':
    #     save_case_mean_std(out_path, normalized=True, post_process=post_process)

    prev_step = 'load'
    data_dir = os.path.join(out_path, prev_step + '.mat')
    data = scipy.io.loadmat(data_dir)
    data_new = {}
    for case_name in data.keys():
        if 'Case' not in case_name:
            continue
        data_new[case_name] = {
            'image': data[case_name]['image'][0, 0]
        }
    data = data_new


    for case_name in data.keys():
        image = data[case_name]['image']
        label = copy.deepcopy(image) * 0 + 1
        weak_label = copy.deepcopy(image) * 0 + 1

        if normalized:
            # if Params['data_name'] == 'Promise':
            #     mean_std = {'mean': image.mean(), 'std': image.std()}
            if Params['data_name'] == 'Segthor':
                image[image < clip_window[0]] = clip_window[0]
                image[image > clip_window[1]] = clip_window[1]
            image = norm_arr(image)
            # image = image.astype(np.float32)
            # image = (image - mean_std['mean']) / mean_std['std']
            # if Params['data_name'] == 'Promise':
            #     image[image < -3] = -3  # normal distribution
            #     image[image > 3] = 3
            # image = (image - image.min()) / (image.max() - image.min()) * 255.0

        nonzero = np.nonzero(label)
        z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
        weak_label[:z0] = 0
        weak_label[z1+1:] = 0

        count = 0

        for z in range(image.shape[0]):
            image_name = f"{case_name}_{0}_{z:04d}.png"
            # image_name = case_name + '_' + '%04d' % count
            print(image_name, image[z].min(), image[z].max(), image.shape)

            image_z = copy.deepcopy(image[z])
            label_z = copy.deepcopy(label[z])
            weak_label_z = copy.deepcopy(weak_label[z])
            

            box_z = copy.deepcopy(label[z])
            if (box_z == 1).any():
                nonzero = np.nonzero((box_z == 1).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                box_z[x0:x1, y0:y1] = 1

            thickbox_z = copy.deepcopy(weak_label[z])
            if (thickbox_z == 255).any() and (thickbox_z == 0).any():
                nonzero = np.nonzero((thickbox_z == 255).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                thickbox_z[x0:x1, y0:y1] = 1

            resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
            image_z: np.ndarray = resize_(image_z, shape).astype(np.uint8)
            label_z: np.ndarray = resize_(label_z, shape).astype(np.uint8)
            box_z: np.ndarray = resize_(box_z, shape).astype(np.uint8)
            thickbox_z: np.ndarray = resize_(thickbox_z, shape).astype(np.uint8)
            # weak_label_z: np.ndarray = resize_(weak_label_z, shape).astype(np.uint8)
            assert 0 <= image_z.min() and image_z.max() <= 255  # The range might be smaller
            assert set(uniq(label_z)).issubset(set(uniq(label)))
            assert set(uniq(box_z)).issubset(set(uniq(label)))
            assert set(uniq(thickbox_z)).issubset(set(uniq(label))) or set(uniq(thickbox_z)).issubset(set([255]))
            # assert set(uniq(weak_label_z)).issubset(set(uniq(weak_label)))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                phase = 'train'
                imsave(os.path.join(sub_dirs[phase]['img'], image_name), image_z)
                imsave(os.path.join(sub_dirs[phase]['gt'], image_name), label_z)
                imsave(os.path.join(sub_dirs[phase]['box'], image_name), box_z)
                imsave(os.path.join(sub_dirs[phase]['thickbox'], image_name), thickbox_z)
                phase = 'val'
                imsave(os.path.join(sub_dirs[phase]['img'], image_name), image_z)
                imsave(os.path.join(sub_dirs[phase]['gt'], image_name), label_z)
                imsave(os.path.join(sub_dirs[phase]['box'], image_name), box_z)
                imsave(os.path.join(sub_dirs[phase]['thickbox'], image_name), thickbox_z)

            count += 1
    check_data_for_midl(out_path, subdir='2d_midl') # subdir='2d_midl-Aug' or '2d_midl'
    ipdb.set_trace()


def save_oritxt_for_kernelcut():
    # txt_dir = '/group/lishl/weak_datasets/Promise12/processed_train_weak_percent_1.0_random_kernelcut/2d_norm/splits'
    txt_dir = '/group/lishl/weak_datasets/0108_SegTHOR/processed_trachea_train_weak_percent_1.0_random_kernelcut/2d_norm/splits'
    train_dir = os.path.join(txt_dir, 'train_fg.txt')
    sampled_dirs = {x: os.path.join(txt_dir, 'train_fg_{}.txt'.format(str(x))) for x in [1.0, 0.8, 0.6, 0.4, 0.2]}
    train_f = open(train_dir, 'r')
    sample_fs = {x: open(sampled_dirs[x], 'w') for x in sampled_dirs.keys()}
    train_all = train_f.readlines()
    for line in train_all:
        sample_fs[1.0].write(line)
        tmp = random.random()
        if tmp < 0.8:
            sample_fs[0.8].write(line)
            if tmp < 0.6:
                sample_fs[0.6].write(line)
                if tmp < 0.4:
                    sample_fs[0.4].write(line)
                    if tmp < 0.2:
                        sample_fs[0.2].write(line)
    for _, f in sample_fs.items():
        f.close()
    train_f.close()
    ipdb.set_trace()


def load_3d_to_2d_kernelcut(out_path, Params, normalized=False, post_process=''):
    error_count = 0
    folders = ['image', 'label', 'weak_label', 'splits', 'visualize']
    subdir = '2d'
    if normalized:
        subdir += '_norm'
    if 'kernelcut' in post_process:
        subdir += '_nobg'
    save_2d_dir = os.path.join(out_path, subdir)
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    sub_dirs = {folder: os.path.join(save_2d_dir, folder) for folder in folders}
    for _, sub_dir in sub_dirs.items():
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    ori_splits_dirs = {phase: os.path.join(out_path, 'splits', phase + '.txt') for phase in ['train', 'val']}
    ori_splits = {phase: [] for phase in ['train', 'val']}
    for phase, ori_split_dir in ori_splits_dirs.items():
        with open(ori_split_dir, 'r') as f:
            ori_splits[phase] = f.readlines()
            ori_splits[phase] = [line.strip() for line in ori_splits[phase]]

    slice_split_types = ['train_' + slice_type for slice_type in ['fg', 'bg', 'fgbg', 'fgbgun']] + ['val']
    split_file_dirs = {phase: os.path.join(sub_dirs['splits'], phase + '.txt') for phase in slice_split_types}
    split_files = {phase: open(split_file_dirs[phase], 'w') for phase in slice_split_types}

    if Params['data_name'] == 'Segthor':
        mean_std = {'mean': 20.78, 'std': 180.50}
        clip_window = [-986, 271]
    if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
        save_case_mean_std(out_path, normalized=True, post_process=post_process)

    prev_step = 'load'
    data = load_data(prev_step)
    for case_name in data.keys():
        image = data[case_name]['image']
        label = data[case_name]['label']
        weak_label = data[case_name]['weak_label']

        if normalized:
            if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
                mean_std = {'mean': image.mean(), 'std': image.std()}
            if Params['data_name'] == 'Segthor':
                image[image < clip_window[0]] = clip_window[0]
                image[image > clip_window[1]] = clip_window[1]
            image = image.astype(np.float32)
            image = (image - mean_std['mean']) / mean_std['std']
            if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
                image[image < -3] = -3  # normal distribution
                image[image > 3] = 3
            image = (image - image.min()) / (image.max() - image.min()) * 255.0

        nonzero = np.nonzero(label)
        z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
        weak_label[:z0] = 0
        weak_label[z1+1:] = 0

        count = 0
        if case_name in ori_splits['train']:
            phase = 'train'
        elif case_name in ori_splits['val']:
            phase = 'val'
        else:
            raise NotImplementedError
        for z in range(image.shape[0]):
            image_name = case_name + '_' + '%04d' % count
            print(image_name, image[z].min(), image[z].max(), image.shape)

            weak_label_z = copy.deepcopy(weak_label[z])
            if 'kernelcut' in post_process and (weak_label_z == 0).any() and (weak_label_z == 255).any():
                nonzero = np.nonzero((weak_label_z == 255).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                weak_label_z[:x0 - 3, :] = 255
                weak_label_z[x1 + 3:, :] = 255
                weak_label_z[:, :y0 - 3] = 255
                weak_label_z[:, y1 + 3:] = 255

            scipy.io.savemat(os.path.join(sub_dirs['image'], image_name + '.mat'), {'image': image[z]})
            scipy.io.savemat(os.path.join(sub_dirs['label'], image_name + '.mat'), {'label': label[z]})
            scipy.io.savemat(os.path.join(sub_dirs['weak_label'], image_name + '.mat'), {'weak_label': weak_label_z})
            if int(case_name[-2:]) < 10:
                visualize_2d(image_name, image[z], label[z], weak_label_z, sub_dirs['visualize'])
            if phase == 'val':
                split_files[phase].write(image_name + '\n')
            else:
                if (weak_label[z] != 0).sum() == 0:
                    split_files['train_bg'].write(image_name + '\n')
                    split_files['train_fgbg'].write(image_name + '\n')
                    split_files['train_fgbgun'].write(image_name + '\n')
                elif (weak_label[z] == 1).sum() > 0:
                    split_files['train_fg'].write(image_name + '\n')
                    split_files['train_fgbg'].write(image_name + '\n')
                    split_files['train_fgbgun'].write(image_name + '\n')
                elif (weak_label[z] == 0).sum() == 0 and (weak_label[z] == 1).sum() == 0:
                    split_files['train_fgbgun'].write(image_name + '\n')
                else:
                    error_count += 1
                    split_files['train_fgbgun'].write(image_name + '\n')
            count += 1
    for _, f in split_files.items():
        f.close()
    print('error count', '-'*10, error_count)
    ipdb.set_trace()


def load_3d_to_2d_kernelcut_test(out_path, Params, normalized=False, post_process=''):
    # out_path = '/group/lishl/weak_datasets/LA_dataset/1011_test_weak_percent_1.0'
    out_path = '/group/lishl/weak_datasets/0108_SegTHOR/1011_test_weak_percent_1.0'
    error_count = 0
    folders = ['image', 'label', 'weak_label', 'splits', 'visualize']
    subdir = '2d'
    if normalized:
        subdir += '_norm'
    if 'kernelcut' in post_process:
        subdir += '_nobg'
    save_2d_dir = os.path.join(out_path, subdir)
    if not os.path.exists(save_2d_dir):
        os.makedirs(save_2d_dir)
    sub_dirs = {folder: os.path.join(save_2d_dir, folder) for folder in folders}
    for _, sub_dir in sub_dirs.items():
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    slice_split_types = ['train_' + slice_type for slice_type in ['fg', 'bg', 'fgbg', 'fgbgun']] + ['val']
    split_file_dirs = {phase: os.path.join(sub_dirs['splits'], phase + '.txt') for phase in slice_split_types}
    split_files = {phase: open(split_file_dirs[phase], 'w') for phase in slice_split_types}

    prev_step = 'load'
    data_dir = os.path.join(out_path, prev_step + '.mat')
    data = scipy.io.loadmat(data_dir)
    data_new = {}
    for case_name in data.keys():
        if 'Case' not in case_name:
            continue
        data_new[case_name] = {
            'image': data[case_name]['image'][0, 0]
        }
    data = data_new

    if Params['data_name'] == 'Segthor':
        mean_std = {'mean': 20.78, 'std': 180.50}
        clip_window = [-986, 271]
    if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
        save_case_mean_std(out_path, normalized=True, post_process=post_process, data=data)

    for case_name in data.keys():
        image = data[case_name]['image']
        label = copy.deepcopy(image) * 0 + 1
        weak_label = copy.deepcopy(image) * 0 + 1

        if normalized:
            if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
                mean_std = {'mean': image.mean(), 'std': image.std()}
            if Params['data_name'] == 'Segthor':
                image[image < clip_window[0]] = clip_window[0]
                image[image > clip_window[1]] = clip_window[1]
            image = image.astype(np.float32)
            image = (image - mean_std['mean']) / mean_std['std']
            if Params['data_name'] == 'Promise' or Params['data_name'] == 'LA':
                image[image < -3] = -3  # normal distribution
                image[image > 3] = 3
            image = (image - image.min()) / (image.max() - image.min()) * 255.0

        nonzero = np.nonzero(label)
        z0, z1 = nonzero[0].min(), nonzero[0].max()     # z axis=0
        weak_label[:z0] = 0
        weak_label[z1+1:] = 0

        count = 0
        for z in range(image.shape[0]):
            image_name = case_name + '_' + '%04d' % count
            print(image_name, image[z].min(), image[z].max(), image.shape)

            weak_label_z = copy.deepcopy(weak_label[z])
            if 'kernelcut' in post_process and (weak_label_z == 0).any() and (weak_label_z == 255).any():
                nonzero = np.nonzero((weak_label_z == 255).astype(int))
                x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max() + 1, nonzero[1].min(), nonzero[1].max() + 1
                weak_label_z[:x0 - 3, :] = 255
                weak_label_z[x1 + 3:, :] = 255
                weak_label_z[:, :y0 - 3] = 255
                weak_label_z[:, y1 + 3:] = 255

            scipy.io.savemat(os.path.join(sub_dirs['image'], image_name + '.mat'), {'image': image[z]})
            scipy.io.savemat(os.path.join(sub_dirs['label'], image_name + '.mat'), {'label': label[z]})
            scipy.io.savemat(os.path.join(sub_dirs['weak_label'], image_name + '.mat'), {'weak_label': weak_label_z})
            if int(case_name[-2:]) < 85:
                visualize_2d(image_name, image[z], label[z], weak_label_z, sub_dirs['visualize'])
            split_files['val'].write(image_name + '\n')
            split_files['train_fg'].write(image_name + '\n')
            split_files['train_bg'].write(image_name + '\n')
            split_files['train_fgbg'].write(image_name + '\n')
            split_files['train_fgbgun'].write(image_name + '\n')
            count += 1
    for _, f in split_files.items():
        f.close()
    print('error count', '-'*10, error_count)
    ipdb.set_trace()


def visualize_2d(image_name, image, label, weak_label, save_vis_dir, additional=None):
    fig = plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.set_title('image_' + str(image.min()) + '_' + str(image.max()))
    plt.imshow(image)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('label_' + str(label.min()) + '_' + str(label.max()))
    plt.imshow(label)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('weak_label_' + str(weak_label.min()) + '_' + str(weak_label.max()))
    plt.imshow(weak_label)
    if additional is not None:
        ax = plt.subplot(2, 2, 4)
        ax.set_title('additional_' + str(additional.min()) + '_' + str(additional.max()))
        plt.imshow(additional)
    plt.savefig(save_vis_dir + '/{}.png'.format(image_name))
    plt.close('all')


def save_case_mean_std(out_path, normalized=False, post_process='', data=None):
    prev_step = 'load'
    subdir = '2d'
    if normalized:
        subdir += '_norm'
    if 'kernelcut' in post_process:
        subdir += '_nobg'
    mean_std_dir = os.path.join(out_path, subdir, 'splits')
    if not os.path.exists(mean_std_dir):
        os.makedirs(mean_std_dir)
    save_mean_std_dir = os.path.join(mean_std_dir, 'case_mean_std.json')
    if data is None:
        data = load_data(prev_step)

    mean_std_all = {}
    for case_name in data.keys():
        image = data[case_name]['image']

        if data_name in ['Promise', 'promise', 'LA']:
            image_value = image#[image != 0]
            mean = image_value.mean()
            std = image_value.std()
            mean_std_all[case_name] = {'mean': mean, 'std': std}

    with open(save_mean_std_dir, 'w') as f:
        json.dump(mean_std_all, f, cls=NpEncoder)


if __name__ == '__main__':
    # save_oritxt_for_kernelcut() # temporary
    '''
    Instructions: 
        If to use random selection, set random_select > 0. 
        e.g. random_select = 3, 
        including the first and the last slices, and a random slice in between.
        If to use uniform selection, set random_select = -1 and weak_stride > 0.
        e.g. weak_stride = 2, 
        including the first and the last slices, and every second slice since the first till the last.
    
    Datasets params:
        1.Promise: binary classes [0,1]
        2.Segthor: {1: esophagus, 2: heart, 3: trachea, 4:aorta}
    '''

    # general datasets setting
    datasets_info = {
        'Promise': {
            'target_spacing': [2.2, 0.61, 0.61],
            'data_dir': '/group/lishl/weak_datasets/Promise12',
        },

        'Segthor': {
            'target_spacing': [2.50, 0.89, 0.89],
            'data_dir': '/group/lishl/weak_datasets/0108_SegTHOR',
        },

        'LA': {
            'target_spacing': [1, 1, 1],
            'data_dir': '/group/lishl/weak_datasets/LA_dataset/'
        },
    }
    organ_name = None
    organ_label = None

    # 0.Params to be set:
    Params = {
        'stage_phase':      1,            # {0: generate weak labels, 1: process unified weak dataset}
        'data_name':        'LA',    # {'Segthor', 'Promise', 'LA'}
        'organ_name':       '',           # {'esophagus', 'heart', 'trachea', 'aorta'}'
        'train_phase':      'train',      # {'train', 'test'}

        'weak':             True,
        'n_percent':        1.0,          # {1.0, 0.5, 0.3, 0.1}
        'sample_style':     'random',     # {'uniform', 'random'}
        'to_kernelcut':     False,         # {False: cross_scribble+tight box, True: fg+bg scribbles}
        'to_scribble':      False,        # {True: fg axis + bg scribble (by dilation)}
        'to_tightbox':      False,        # {True: only generate tight bbox}
        'to_point':         False,        # {True: fg point (loose_box_dist), bg point (inner_dist)}

        'ref_dir':          '',    # 1011_train_weak_percent_0.5_random  for Kernelcut, Reference weak label dir, from which load and modify label
        'all_percents':     [1.0, 0.5, 0.3, 0.1],    # r1 in r3, r3 in r5 setting.

        'fg_type':          'major_axis', # {'major_axis', 'scribble', 'point'}, sribble:cross lines
        'bg_type':          'loose',      # {'', 'loose'}, '': default tight box
        'line_width':       3,            # {3, 4}
        'boundary_margin':  2,            # 2, current 5 only for point
        'loose_box_dist':   [20, 50],     # [10, 20]
        'inner_dist':       5,            # only used for 'point labels'
        'margin_ratio':     0.1,          # expand largest box ratio: along xy plane
        'max_memory':       128 * 128 * 128,

        'dirprefix':        '1011_',      # '1009_', '1011_', '1012_', '1020_', '1024_', '1111_'
        'dirpostfix':       '',

        'post_process':     'kernelcut',  # {'': fg scribble + bg region, 'kernelcut': fg+bg scribbles}
    }
    print(Params)
    to_kernelcut = Params['to_kernelcut']
    to_scribble = Params['to_scribble']
    to_tightbox = Params['to_tightbox']
    to_point = Params['to_point']

    # 1.dataset params: set {data_name || organ_name}
    data_name = Params['data_name']        # 'Segthor', 'Promise'
    data_info = datasets_info[data_name]
    if data_name in ['Segthor']:
        organ_dict = {'esophagus': 1, 'heart': 2, 'trachea': 3, 'aorta': 4}
        organ_name = Params['organ_name']    #'trachea'
        organ_label = organ_dict[organ_name]
        depth_transpose = True
    elif data_name in ['LA']:
        depth_transpose = True

    # 2.weak label params: set {weak || weak_stride || random_select}
    weak = Params['weak']
    n_percent = Params['n_percent']          # {0.05, 0.1, 0.2, 0.3, 0.5}
    sample_style = Params['sample_style']    # {'uniform', 'random}
    margin_ratio = Params['margin_ratio']    # for prior crop
    memory = Params['max_memory']
    line_width = Params['line_width']
    boundary_margin = Params['boundary_margin']

    # 3.downsample params: set {downsample manner || use_postfix}
    downsample_dict = {'Z': 1 , 'XY': 2, 'XYZ': 3}
    downsample_manner = 'XY'                 # 'XYZ' #
    use_postfix = True
    changed_axis = downsample_dict[downsample_manner]
    if use_postfix and changed_axis != 2:
        postfix = '_dz' if changed_axis == 1 else '_dxyz'
    else:
        postfix = ''
    dirprefix = Params['dirprefix']                     # 'processed_'

    # automatic pipeline
    phase = Params['train_phase']
    if organ_name is not None:
        phase = organ_name + '_' + phase
    root = data_info['data_dir']
    out_path = os.path.join(root, dirprefix + phase)
    re_sample = data_info['target_spacing']

    ## Path: phase 0
    phase0_path = os.path.join(root, dirprefix + phase + '_weak_{}'.format(sample_style))
    if to_kernelcut:
        phase0_path = phase0_path + '_kernelcut'
    if to_scribble:
        phase0_path = phase0_path + '_scribble'
    if to_tightbox:
        phase0_path = phase0_path + '_tightbox'
    if to_point:
        phase0_path = phase0_path + '_point'

    ## Path: phase 0 or 1
    if Params['stage_phase'] == 0:          # generate all weak labels
        print('Enter phase 0: generate all weak labels! (n_percent=1.0)')
        if n_percent != 1.0:
            n_percent = 1.0
        out_path = phase0_path
    else:
        if weak:
            out_path = os.path.join(root, dirprefix + phase + '_weak_percent_{}_{}'.format(n_percent, sample_style))
        if use_postfix:
            out_path = out_path + postfix
        if to_kernelcut:
            out_path = out_path + '_kernelcut'
        if to_scribble:
            out_path = out_path + '_scribble'
        if to_tightbox:
            out_path = out_path + '_tightbox'
        if to_point:
            out_path = out_path + '_point'
        if Params['dirpostfix'] != '':
            out_path = out_path + Params['dirpostfix']
    ## Path: make dirs
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_split_dir = os.path.join(out_path, 'splits')
    if not os.path.exists(save_split_dir):
        os.makedirs(save_split_dir)

    # TODO: KernelCut process
    # save_case_mean_std(out_path, normalized=False)
    # ipdb.set_trace()
    # load_3d_to_2d_kernelcut(out_path, Params, normalized=True, post_process=Params['post_process'])
    # ipdb.set_trace()
    # load_3d_to_2d_kernelcut_test(out_path, Params, normalized=True, post_process=Params['post_process'])
    # ipdb.set_trace()
    # load_3d_to_2d_midlbox(out_path, Params, normalized=True, post_process='midl') # then aug, then visualize
    # ipdb.set_trace()

    # load_3d_to_2d_midlbox_test(out_path, Params, normalized=True, post_process='midl') # then aug, then visualize
    # ipdb.set_trace()
    # check_data_for_midl(out_path, subdir='2d_midl-Aug') # subdir='2d_midl-Aug' or '2d_midl'
    # ipdb.set_trace()

    # pipe1: phase 0, generate weak labels then save; phase 1, load selected weak labels
    if Params['stage_phase'] == 0:
        if data_name in ['Promise']:
            load_mhd(root, phase, out_path, weak, n_percent=n_percent, sample_style=sample_style,
                     line_width=line_width, boundary_margin=boundary_margin)
        elif data_name in ['Segthor']:
            load_nii(root, phase, out_path, weak, n_percent=n_percent, sample_style=sample_style,
                     line_width=line_width, boundary_margin=boundary_margin,
                     organ_label=organ_label, depth_transpose=depth_transpose)
        elif data_name in ['LA']:
            load_la(root, phase, out_path, weak, n_percent=n_percent, sample_style=sample_style,
                     line_width=line_width, boundary_margin=boundary_margin, depth_transpose=depth_transpose)

        get_train_val_split(save_split_dir, data_name=data_name)
        # log write readme.txt
        with open(os.path.join(out_path, 'readme.txt'), 'w') as f:
            f.write('Params:\n')
            for pk, pv in Params.items():
                f.write('%s:\t%s\n' % (str(pk), str(pv)))
        print('All weak labels saved along with images and gts!')
        sys.exit(); ipdb.set_trace()
    else:
        pass
        load_saved(phase0_path, out_path, weak, n_percent=n_percent, ref_dir=Params['ref_dir'])
        get_train_val_split(save_split_dir, data_name=data_name)
    ipdb.set_trace()    # rollback
    resample(out_path)
    align_volume_center(out_path)
    zoom_factor, roi, down_n_all = crop_roi(out_path, margin_ratio, memory, weak, params_only=False, changed_axis=changed_axis)

    print('crop_roi params', zoom_factor, roi, down_n_all)
    save_params({'zoom_factor': zoom_factor, 'roi': roi, 'down_n_all': down_n_all}, save_split_dir)

    normalize_intensity(out_path, data_name=data_name)
    expand_bg(out_path)

    final_spacing = compute_final_spacing(re_sample, zoom_factor)
    num_slices, num_labeled_slices = compute_num_labeled_slices(out_path)


    save_params({
        'datetime': str(datetime.datetime.now()),
        'weak': weak,
        'n_percent': n_percent,
        'sample_style': sample_style,
        'num_slices': num_slices,
        'num_labeled_slices': num_labeled_slices,
        'line_width': line_width,
        'boundary_margin': boundary_margin,
        're_sample': re_sample,
        'margin_ratio': margin_ratio,
        'memory': memory,
        'interest_label': organ_label,
        'phase': phase,
        'root': root,
        'out_path': out_path,
        'down_sample_ratios': zoom_factor,
        'down_n_all': down_n_all,
        'final_spacing': final_spacing,
        'roi': roi,
        'crop_shape': [roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
    }, save_split_dir)

    # graphcut process
    if weak and to_tightbox:
        modify_bg(out_path)
        graphcut_label(out_path)
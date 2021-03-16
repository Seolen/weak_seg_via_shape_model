from batchgenerators.dataloading import SlimDataLoaderBase
from glob import glob
import os
import numpy as np
from scipy import io
import nibabel as nib
from scipy.io import loadmat, savemat
import copy
from skimage.segmentation import find_boundaries
import scipy
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening, generate_binary_structure
import random

import ast
import ipdb

def gen_extra(self, arr, extra_near_nums, extra_far_nums, extra_near_dist, extra_radius, extra_slices):
    # extraneous separate region: near branch point
    # 0. find index of branch point
    branch_index = -1
    ulabel = None
    for index in range(arr.shape[0] - 2, 0, -1):
        ulabel1, num_features1 = scipy.ndimage.label(arr[index])
        ulabel2, num_features2 = scipy.ndimage.label(arr[index + 1])
        if (num_features1 + num_features2) == 3:
            branch_index = index if num_features1 == 1 else index + 1
            ulabel = ulabel1 if num_features1 == 1 else ulabel2
            break
    # 1. determine center
    if (ulabel is not None) and (ulabel.sum() > 0):
        _nonzero = np.nonzero(ulabel)
        _x0, _x1, _y0, _y1 = _nonzero[0].min(), _nonzero[0].max(), _nonzero[1].min(), _nonzero[1].max()
        _xcenter, _ycenter = (_x0 + _x1) // 2, (_y0 + _y1) // 2
        range_x = [max(0, _xcenter - extra_near_dist[1]), min(arr.shape[1] - 1, _xcenter + extra_near_dist[1])]
        range_y = [max(0, _ycenter - extra_near_dist[2]), min(arr.shape[2] - 1, _ycenter + extra_near_dist[2])]
        range_z = [max(0, branch_index - extra_near_dist[0]), min(arr.shape[0] - 1, branch_index + extra_near_dist[0])]

        arr_near_holes = np.zeros_like(arr, dtype=arr.dtype)
        arr_near_holes[range_z[0]: range_z[1], range_x[0]: range_x[1], range_y[0]: range_y[1]] = 1
        # # inner roi
        # arr_near_holes[arr==1] = 0
        # on boundary
        arr_near_holes[range_z[0 ] +1: range_z[1 ] -1, range_x[0 ] +1: range_x[1 ] -1, range_y[0 ] +1: range_y[1 ] -1] = 0
        nonzero = np.nonzero(arr_near_holes)
        length = len(nonzero[0])

        # # another esophagus simulation
        # xs = list(range(max(0, _xcenter - 10), min(arr.shape[1] - 1, _xcenter + 10)))
        # ys = list(range(max(0, _ycenter - 25), max(0, _ycenter - 15))) + \
        #           list(range(max(0, _ycenter + 15), max(0, _ycenter + 25)))

        for ith in range(random.choice(list(range(extra_near_nums[0], extra_near_nums[1] + 1)))):
            rand_index = random.choice(list(range(length)))
            _z, _x, _y = nonzero[0][rand_index], nonzero[1][rand_index], nonzero[2][rand_index]
            # # another simulation
            # _z = random.choice(list(range(range_z[0], range_z[1])))
            # _x = random.choice(xs); _y = random.choice(ys)

            center = [_z, _x, _y]

            # 2. near branch point: dilation
            arr2 = np.zeros_like(arr, dtype=arr.dtype)
            arr2[center[0], center[1], center[2]] = 1
            _iter = random.choice(list(range(extra_radius[0], extra_radius[1 ] +1)))
            arr2[center[0]] = binary_dilation(arr2[center[0]], iterations=_iter).astype(arr.dtype)
            extra_slice_num = random.choice(list(range(extra_slices[0], extra_slices[1] + 1)))
            for ith in range(max(0, center[0] - extra_slice_num // 2), min(arr.shape[0 ] -1, center[0 ] +extra_slice_num//2)):
                arr2[ith] = arr2[center[0]]
            arr[arr2 == 1] = 1
    else:
        # no branch point exist, set object center
        _nonzero = np.nonzero(arr)
        _z0, _z1, _x0, _x1, _y0, _y1 = _nonzero[0].min(), _nonzero[0].max(), _nonzero[1].min(), _nonzero[1].max(), \
                                       _nonzero[2].min(), _nonzero[2].max()
        _zcenter, _xcenter, _ycenter = (_y0 + _y1) // 2, (_x0 + _x1) // 2, (_y0 + _y1) // 2
        range_x = [max(0, _xcenter - extra_near_dist[1]), min(arr.shape[1] - 1, _xcenter + extra_near_dist[1])]
        range_y = [max(0, _ycenter - extra_near_dist[2]), min(arr.shape[2] - 1, _ycenter + extra_near_dist[2])]
        range_z = [max(0, _zcenter - extra_near_dist[0]), min(arr.shape[0] - 1, _zcenter + extra_near_dist[0])]

    # 3. far branch point: dilation
    candidate = np.ones_like(arr, dtype=np.uint8)
    candidate[range_z[0]: range_z[1], range_x[0]: range_x[1], range_y[0]: range_y[1]] = 0
    candidate[:range_z[0], ...] = 0
    candidate[range_z[1]:, ...] = 0
    nonzero = np.nonzero(candidate)
    length = len(nonzero[0])

    for ith in range(random.choice(list(range(extra_far_nums[0], extra_far_nums[1] + 1)))):
        rand_index = random.choice(list(range(length)))
        _z, _x, _y = nonzero[0][rand_index], nonzero[1][rand_index], nonzero[2][rand_index]
        center = [_z, _x, _y]

        # 4. near region: dilation
        arr2 = np.zeros_like(arr, dtype=arr.dtype)
        arr2[center[0], center[1], center[2]] = 1
        _iter = random.choice(list(range(extra_radius[0], extra_radius[1])))
        arr2[center[0]] = binary_dilation(arr2[center[0]], iterations=_iter).astype(arr.dtype)
        extra_slice_num = random.choice(list(range(extra_slices[0], extra_slices[1] + 1)))
        for ith in range(max(0, center[0] - extra_slice_num // 2),
                         min(arr.shape[0] - 1, center[1] + extra_slice_num // 2)):
            arr2[ith] = arr2[center[0]]
        arr[arr2 == 1] = 1

    return arr

#############################################
    ''' 2D dilate
    # 2d dilation and copy the same operation
    z_index = nonzero[0][index]
    seeds[z_index] = binary_dilation(seeds[z_index], iterations=dilate_iter).astype(seeds.dtype)
    ref_slice = seeds[z_index]
    slice_num = random.choice(list(range(dilate_slices[0], dilate_slices[1] + 1)))
    start_index, end_index = z0, z1
    if z_index - start_index < slice_num//2:
        end_index = min(start_index + slice_num, z1)
    elif end_index - z_index < slice_num//2:
        start_index = max(end_index - slice_num, z0)
    else:
        start_index = max(z_index - slice_num//2, z0)
        end_index = min(z_index + slice_num//2, z1)
    
    for _slice in range(start_index, end_index+1):
        seeds[_slice][ref_slice==1] = 1
    arr[seeds == 1] = 1
    '''

    '''
    # 2d erode
    z_index = nonzero[0][index]
    struct = generate_binary_structure(2, 2)
    seeds[z_index] = binary_erosion(seeds[z_index], structure=struct, iterations=erode_iter).astype(seeds.dtype)
    ref_slice = seeds[z_index]
    slice_num = random.choice(list(range(erode_slices[0], erode_slices[1] + 1)))
    start_index, end_index = z0, z1
    if z_index - start_index < slice_num//2:
        end_index = min(start_index + slice_num, z1)
    elif end_index - z_index < slice_num//2:
        start_index = max(end_index - slice_num, z0)
    else:
        start_index = max(z_index - slice_num//2, z0)
        end_index = min(z_index + slice_num//2, z1)

    for _slice in range(start_index, end_index+1):
        seeds[_slice][ref_slice==0] = 0
    arr[seeds == 0] = 0
    '''
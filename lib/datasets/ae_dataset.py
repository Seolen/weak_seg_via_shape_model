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
from lib.datasets.lib_sample import trachea_libs, prostate_libs, la_libs
from lib.datasets.lib_sample_scribble import trachea_libs_scribble, prostate_libs_scribble, la_libs_scribble
from lib.datasets.lib_sample_scribble_far import trachea_libs_scribble_far, prostate_libs_scribble_far, la_libs_scribble_far, \
    trachea_libs_scribble_s, trachea_libs_scribble_t, prostate_libs_scribble_s, prostate_libs_scribble_t, la_libs_scribble_s, \
    la_libs_scribble_t, trachea_libs_tightbox, prostate_libs_tightbox
from lib.configs.parse_arg import opt, args

'''
train_dir: /group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_train/
val_dir:   /group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_val/
'''


class AEDataset(SlimDataLoaderBase):
    def __init__(self, train_dir, val_dir, batch_size, phase='train', split_file=None,
                 use_weak=False,
                 shuffle=False, seed_for_shuffle=None, infinite=False, return_incomplete=False,
                 num_threads_in_multithreaded=1,
                 noise_params=None, use_pred_num=5, use_prob_top=True,
                 pseudo_label_lcc=False, label_percent=30, anisotropy=False,
                 val_mode=False, use_train_all=False):
        """
        :param data: datadir or data dictionary
        :param batch_size:
        :param batch_size: dataset name, 'Prostate' or 'Heart'
        :param split_file: 'train.txt' or 'val.txt'
        :param shuffle: return shuffle order or not
        :param trans_depth: tranpose depth dimension to the second dim, e.g. (B, W, H, D) -> (B, D, W, H)

        :param use_lcc： use largest connected component method, when loading fixed training preds as pseudo label

        Each iteration: return {'data': ,'seg': }, with shape (B, W, H, D)
        """
        super(AEDataset, self).__init__(train_dir, batch_size, num_threads_in_multithreaded)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.infinite = infinite
        self.use_weak = use_weak

        self.noise_params = noise_params
        self.pseudo_label_lcc = pseudo_label_lcc
        self.anisotropy = anisotropy
        self.val_mode = val_mode

        if opt.data.dataname in ['Prostate']:
            sample_libs = prostate_libs
        elif opt.data.dataname in ['Trachea']:
            sample_libs = trachea_libs
        elif opt.data.dataname in ['LA']:
            sample_libs = la_libs

        # extension for scribble labels
        if 'k1012_' in train_dir:
            print('AEDataset: Scribble libs loaded.')
            if '_far' not in train_dir:
                if opt.data.dataname in ['Prostate']:
                    sample_libs = prostate_libs_scribble
                elif opt.data.dataname in ['Trachea']:
                    sample_libs = trachea_libs_scribble
                elif opt.data.dataname in ['LA']:
                    sample_libs = la_libs_scribble
            else:
                if opt.data.dataname in ['Prostate']:
                    sample_libs = prostate_libs_scribble_far
                elif opt.data.dataname in ['Trachea']:
                    sample_libs = trachea_libs_scribble_far
                elif opt.data.dataname in ['LA']:
                    sample_libs = la_libs_scribble_far
        elif 's1012_' in train_dir:
            print('AEDataset: Scribble (dilation) libs loaded.')
            if opt.data.dataname in ['Prostate']:
                sample_libs = prostate_libs_scribble_s
            elif opt.data.dataname in ['Trachea']:
                sample_libs = trachea_libs_scribble_s
            elif opt.data.dataname in ['LA']:
                sample_libs = la_libs_scribble_s
        elif 't1012' in train_dir:
            print('AEDataset: Tight box libs loaded.')
            if opt.data.dataname in ['Prostate']:
                sample_libs = prostate_libs_scribble_t
            elif opt.data.dataname in ['Trachea']:
                sample_libs = trachea_libs_scribble_t
            elif opt.data.dataname in ['LA']:
                sample_libs = la_libs_scribble_t
        elif 't0312' in train_dir:
            print('AEDataset: Tight box 0312 libs loaded.')
            if opt.data.dataname in ['Trachea']:
                sample_libs = trachea_libs_tightbox
            elif opt.data.dataname in ['Prostate']:
                sample_libs = prostate_libs_tightbox
            else:
                raise NotImplementedError
                ipdb.set_trace()

        _group = 'top_prob' if use_prob_top else 'upper_bound'
        if phase == 'train':
            if use_train_all:
                self.samples = sample_libs[phase]
            else:
                self.samples = sample_libs['train_select'][label_percent][_group][use_pred_num]
        else:
            self.samples = sample_libs[phase]

        if phase == 'train':
            self.sample_paths = [os.path.join(train_dir, 'mat/%s.mat' % _id) for _id in self.samples]
        else:
            self.sample_paths = [os.path.join(val_dir, 'mat/%s.mat' % _id) for _id in self.samples]

        # inner variables
        self.indices = list(range(len(self.samples)))
        seed_for_shuffle = args.seed
        self.rs = np.random.RandomState(seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.return_incomplete = return_incomplete
        self.last_reached = False
        self.number_of_threads_in_multithreaded = 1

    def __len__(self):
        return len(self.samples) // self.batch_size

    def reset(self):
        assert self.indices is not None
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True
        self.rs.seed(self.rs.randint(0, 999999999))
        if self.shuffle:
            self.rs.shuffle(self.indices)
        self.last_reached = False

    def get_indices(self):
        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=None)

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])
                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and (not self.last_reached or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    def generate_train_batch(self):
        # similar to __getiterm__(index), but not index as params
        indices = self.get_indices()
        data = {'image': [], 'label': [], 'pred': []}
        for index in indices:
            sample = loadmat(self.sample_paths[index])
            image = sample['image']
            if self.phase == 'train' and (not self.val_mode):
                label = sample['pred']
                pred = self.dilate(label.copy(), noise_params=self.noise_params, anisotropy=self.anisotropy).astype(np.uint8)
                if opt.ae.gt_opening:
                    label = self.label_opening(label, iteration=opt.ae.gt_opening_iter)
                if self.pseudo_label_lcc:
                    label = self.getLargestCC(label).astype(np.uint8)
            else:
                label = sample['gt_label']
                pred = sample['pred']

            data['image'].append(np.expand_dims(image, 0))
            data['label'].append(np.expand_dims(label, 0))
            data['pred'].append(np.expand_dims(pred, 0))

        for key, value in data.items():
            data[key] = np.array(value)

        return {'data': data['image'], 'seg': data['label'], 'pred': data['pred']}

    def load_split_file(self, split_file):
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    samples.append(line)
        return samples

    def getLargestCC(self, segmentation):
        labels, num_features = scipy.ndimage.label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC.astype(np.uint8)

    def label_opening(self, label, iteration=10):
        '''
        Mhoporlogical opening operation
        - due to the error mode of excessive predictions, we need cut some extrusive branches
        '''
        label_open = binary_opening(label, iterations=iteration).astype(label.dtype)
        return label_open

    def gen_extra2(self, arr, extra_near_nums, extra_radius, extra_slices):
        # extraneous separate region: no near or far
        arr2 = np.ones_like(arr, dtype=arr.dtype)
        arr2[arr==1] = 0
        nonzero = np.nonzero(arr2)
        length = len(nonzero[0])
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
            _iter = random.choice(list(range(extra_radius[0], extra_radius[1] + 1)))
            arr2[center[0]] = binary_dilation(arr2[center[0]], iterations=_iter).astype(arr.dtype)
            extra_slice_num = random.choice(list(range(extra_slices[0], extra_slices[1] + 1)))
            for ith in range(max(0, center[0] - extra_slice_num // 2),
                             min(arr.shape[0] - 1, center[0] + extra_slice_num // 2)):
                arr2[ith] = arr2[center[0]]
            arr[arr2 == 1] = 1
        return arr

    def dilate(self, arr, noise_params=None, anisotropy=False,
               Params={'extra_near_nums': [0, 2],
                       'extra_slices': [8, 17], 'extra_radius': [2, 5],
                       'dilate_nums': [1, 4], 'dilate_iters': [5, 11], 'extend_nums_head': [1, 9],
                       'extend_nums_branch': [4, 11], 'close_iters': [3, 6],
                       'discard_nums': [0, 0], 'discard_slices': [1, 3],    # new added discarding slices
                       'erode_nums': [1, 2], 'erode_iters': [5, 11], 'erode_slices': [4, 8]}):
        '''
        dilate, copy, close operation
        :param arr: 3d numpy array
        :param anisotropy: False for trachea, True for prostate
        :return:
        '''
        # external params
        if noise_params is not None:
            for key, value in noise_params.items():
                Params[key] = ast.literal_eval(value)
        # params
        extra_near_nums, extra_slices, extra_radius, dilate_nums, dilate_iters, \
            extend_nums_head, extend_nums_branch, close_iters = Params['extra_near_nums'],\
            Params['extra_slices'], Params['extra_radius'], \
            Params['dilate_nums'], Params['dilate_iters'], Params['extend_nums_head'], \
            Params['extend_nums_branch'], Params['close_iters']
        erode_nums, erode_iters, erode_slices = Params['erode_nums'], Params['erode_iters'], Params['erode_slices']

        nonzero = np.nonzero(arr == 1)
        z0, z1 = nonzero[0].min(), nonzero[0].max()
        # 0. extraneous regions
        # if not anisotropy:
        arr = self.gen_extra2(arr, extra_near_nums, extra_radius, extra_slices)

        # 1. random sample a boundary point, and dilate with 5-10 iter
        boundaries = find_boundaries(arr)
        nonzero = np.nonzero(boundaries == 1)
        indexs = len(nonzero[0])  #
        if isinstance(dilate_nums, list) and dilate_nums[1] != 0:
            for idx in range(random.choice(list(range(dilate_nums[0], dilate_nums[1]+1)))):
                index = np.random.choice(list(range(indexs)))
                seeds = np.zeros_like(arr, dtype=arr.dtype)
                seeds[nonzero[0][index], nonzero[1][index], nonzero[2][index]] = 1
                dilate_iter = np.random.choice(np.array(list(range(dilate_iters[0], dilate_iters[1] + 1))))
                if not anisotropy:
                    # 3d dilation
                    seeds = binary_dilation(seeds, iterations=dilate_iter).astype(seeds.dtype)
                    arr[seeds == 1] = 1
                else:
                    # random select a target slice, do continuous dilation along slices
                    target_index = random.choice(list(range(z0, z1 + 1)))
                    min_tindex = target_index-dilate_iter//2
                    target_indexs = range( max(z0, min_tindex), min(z1+1, min_tindex+dilate_iter) )
                    target_quadrant = random.choice(list(range(4)))
                    for target_index in target_indexs:
                        ref_index = random.choice(list(range(z0, z1 + 1)))
                        arr = self.morph_operate(arr, ref_index, target_index, mode='dilate', quadrant=target_quadrant)


        # 2. random erode a boundary point
        if isinstance(erode_nums, list) and dilate_nums[1] != 0:
            for idx in range(random.choice(list(range(erode_nums[0], erode_nums[1]+1)))):
                index = random.choice(list(range(indexs)))
                seeds = np.ones_like(arr, dtype=arr.dtype)
                seeds[nonzero[0][index], nonzero[1][index], nonzero[2][index]] = 0
                erode_iter = np.random.choice(np.array(list(range(erode_iters[0], erode_iters[1] + 1))))
                if not anisotropy:
                    # 3d dilation
                    seeds = binary_dilation(seeds, iterations=erode_iter).astype(seeds.dtype)
                    arr[seeds == 0] = 0
                else:
                    # random select a target slice, do continuous erosion along slices
                    target_index = random.choice(list(range(z0, z1 + 1)))
                    min_tindex = target_index - erode_iter // 2
                    target_indexs = range(max(z0, min_tindex), min(z1 + 1, min_tindex + erode_iter))
                    target_quadrant = random.choice(list(range(4)))
                    for target_index in target_indexs:
                        ref_index = random.choice(list(range(z0, z1 + 1)))
                        arr = self.morph_operate(arr, ref_index, target_index, mode='erode', quadrant=target_quadrant)

        # 3. copy marginal labels, extend to random 1-8 slices
        if isinstance(extend_nums_head, list) and isinstance(extend_nums_branch, list):
            arr2 = np.zeros_like(arr, dtype=arr.dtype)
            nonzero = np.nonzero(arr == 1)
            z0, z1 = nonzero[0].min(), nonzero[0].max()
            if extend_nums_head[1] != 0:
                extend_num_head = random.choice(list(range(extend_nums_head[0], extend_nums_head[1]+1)))
                if extend_num_head > 0:         # simulate FP slices
                    for index in range(z1 + 1, min(arr.shape[0] - 1, z1 + extend_num_head)):
                        arr2[index, ...] = arr[z1 - 1, ...]
                elif extend_num_head < 0:       # simulate FN slices
                    for index in range(z1 + extend_num_head+1, z1+1):
                        arr2[index, ...] = 0
            if extend_nums_branch[1] != 0:
                extend_num_branch = random.choice(list(range(extend_nums_branch[0], extend_nums_branch[1] + 1)))
                if extend_num_branch > 0:  # simulate FP slices
                    for index in range(max(0, z0 - extend_num_branch), z0):
                        arr2[index, ...] = arr[z0+1, ...]
                elif extend_num_branch < 0:  # simulate FN slices
                    for index in range(z0, z0 - extend_num_branch):
                        arr2[index, ...] = 0

            # arr2 = binary_closing(arr2, iterations=3).astype(arr2.dtype)
            arr[arr2 == 1] = 1

        # 4. close operation: narrow gap of separate branches
        if isinstance(close_iters, list) and close_iters[1] != 0:
            close_iter = np.random.choice(np.array(list(range(close_iters[0], close_iters[1]+1))))
            arr = binary_closing(arr, iterations=close_iter).astype(arr.dtype)

        # *. random discard some slices (not near boundary+-3)
        if isinstance(Params['discard_nums'], list) and Params['discard_nums'][1] != 0:
            for idx in range(random.choice(list(range(Params['discard_nums'][0], Params['discard_nums'][1] + 1)))):
                dslice_num = random.choice(list(range(Params['discard_slices'][0], Params['discard_slices'][1] + 1)))
                center_candidates = list(range(z0+3+dslice_num//2, z1-2-dslice_num//2))
                center_candidate = random.choice(center_candidates)
                arr[center_candidate-dslice_num//2:center_candidate-dslice_num//2+dslice_num+1, :, :] = 0

        # # 0. extraneous regions
        # if self.anisotropy:
        #     arr = self.gen_extra2(arr, extra_near_nums, extra_radius, extra_slices)

        return arr

    def morph_operate(self, arr, ref_index, target_index, mode='dilate', quadrant=None):
        '''
        :param arr:  original label arr
        :param ref_slice:  selected ref_slice
        :param mode:  {'dilate', 'erode'}
        :return:  dilated or eroded arr
        '''

        ref_slice = arr[ref_index]
        inter_value = 1 if mode == 'dilate' else 0

        slice_nonzero = np.nonzero(ref_slice)
        if len(slice_nonzero[0]) == 0:
            return arr
        slice_x0, slice_x1, slice_y0, slice_y1 = slice_nonzero[0].min(), slice_nonzero[0].max(), \
                                                 slice_nonzero[1].min(), slice_nonzero[1].max()
        slice_xcenter, slice_ycenter = (slice_x0 + slice_x1) // 2, (slice_y0 + slice_y1) // 2
        slice_empty = np.zeros_like(arr[ref_index], arr.dtype)
        pieces = [slice_empty.copy(), slice_empty.copy(), slice_empty.copy(), slice_empty.copy()]
        target_quadrants = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}

        # B. for each piece, select another slice, insert it as dilation region
        # for ith, piece in enumerate(pieces):
        # modify: random select a piece and another slice. NOT traverse all pieces.
        ith = random.choice(list(range(4)))
        piece = pieces[ith]
        # target_quadrant = random.choice(target_quadrants[ith])
        if quadrant is None:
            target_quadrant = random.choice(list(range(4)))
        else:
            target_quadrant = quadrant

        # B.1 assign piece
        if ith == 0:  # 1st quadrant
            piece[slice_xcenter:, slice_ycenter:] = ref_slice[slice_xcenter:, slice_ycenter:]
        elif ith == 1:  # 2nd quadrant
            piece[:slice_xcenter, slice_ycenter:] = ref_slice[:slice_xcenter, slice_ycenter:]
        elif ith == 2:  # 3rd quadrant
            piece[:slice_xcenter, :slice_ycenter] = ref_slice[:slice_xcenter, :slice_ycenter]
        else:  # 4th quadrant
            piece[slice_xcenter:, :slice_ycenter] = ref_slice[slice_xcenter:, :slice_ycenter]
        # B.2 select target slice, target quadrant
        ref_target_slice = np.zeros_like(arr[target_index], dtype=arr.dtype)
        target_nonzero = np.nonzero(arr[target_index])
        if len(target_nonzero[0]) == 0:
            return arr  # continue
        target_xcenter, target_ycenter = (target_nonzero[0].min() + target_nonzero[0].max()) // 2, \
                                         (target_nonzero[1].min() + target_nonzero[1].max()) // 2
        target_xlen, target_ylen = target_nonzero[0].max() - target_nonzero[0].min(), \
                                   target_nonzero[1].max() - target_nonzero[1].min()

        rot_k = (target_quadrant - ith) % 4 if mode == 'dilate' else (target_quadrant - ith + 2) % 4
        piece_rot = np.rot90(piece, rot_k)
        piece_nonzero = np.nonzero(piece_rot)
        if len(piece_nonzero[0]) == 0:
            return arr  # continue
        piece_x0, piece_x1, piece_y0, piece_y1 = piece_nonzero[0].min(), piece_nonzero[0].max(), \
                                                 piece_nonzero[1].min(), piece_nonzero[1].max()
        # B.3 assign piece to target slice-quadrant
        if target_quadrant == 0:
            # B.3.1 select a candidate fix point, to insert piece
            tx0, tx1, ty0, ty1 = target_xcenter, target_xcenter + target_xlen // 4, \
                                 target_ycenter, target_ycenter + target_ylen // 4
            ref_target_slice[tx0: tx1, ty0: ty1] = arr[target_index, tx0: tx1, ty0: ty1]
            ref_candidates = np.nonzero(ref_target_slice)
            length = len(ref_candidates[0])
            if length == 0:
                return arr  # continue
            point_index = random.choice(list(range(length)))
            candidate = ref_candidates[0][point_index], ref_candidates[1][point_index]

            incre_x = min(piece_x1 - piece_x0 + 1, ref_target_slice.shape[0] - candidate[0])
            incre_y = min(piece_y1 - piece_y0 + 1, ref_target_slice.shape[1] - candidate[1])
            arr[target_index, candidate[0]: candidate[0] + incre_x,
            candidate[1]: candidate[1] + incre_y] \
                [piece_rot[piece_x0: piece_x0 + incre_x, piece_y0: piece_y0 + incre_y] == 1] = inter_value

        elif target_quadrant == 1:
            tx0, tx1, ty0, ty1 = target_xcenter - target_xlen // 4, target_xcenter, \
                                 target_ycenter, target_ycenter + target_ylen // 4
            ref_target_slice[tx0: tx1, ty0: ty1] = arr[target_index, tx0: tx1, ty0: ty1]
            ref_candidates = np.nonzero(ref_target_slice)
            length = len(ref_candidates[0])
            if length == 0:
                return arr  # continue
            point_index = random.choice(list(range(length)))
            candidate = ref_candidates[0][point_index], ref_candidates[1][point_index]

            decre_x = min(piece_x1 - piece_x0 + 1, candidate[0])
            incre_y = min(piece_y1 - piece_y0 + 1, ref_target_slice.shape[1] - candidate[1])
            arr[target_index, candidate[0] - decre_x: candidate[0],
            candidate[1]: candidate[1] + incre_y] \
                [piece_rot[piece_x1 - decre_x: piece_x1, piece_y0: piece_y0 + incre_y] == 1] = inter_value
        elif target_quadrant == 2:
            tx0, tx1, ty0, ty1 = target_xcenter - target_xlen // 4, target_xcenter, \
                                 target_ycenter - target_ylen // 4, target_ycenter
            ref_target_slice[tx0: tx1, ty0: ty1] = arr[target_index, tx0: tx1, ty0: ty1]
            ref_candidates = np.nonzero(ref_target_slice)
            length = len(ref_candidates[0])
            if length == 0:
                return arr  # continue
            point_index = random.choice(list(range(length)))
            candidate = ref_candidates[0][point_index], ref_candidates[1][point_index]

            decre_x = min(piece_x1 - piece_x0 + 1, candidate[0])
            decre_y = min(piece_y1 - piece_y0 + 1, candidate[1])
            arr[target_index, candidate[0] - decre_x: candidate[0],
            candidate[1] - decre_y: candidate[1]] \
                [piece_rot[piece_x1 - decre_x: piece_x1, piece_y1 - decre_y: piece_y1] == 1] = inter_value
        else:
            tx0, tx1, ty0, ty1 = target_xcenter, target_xcenter + target_xlen // 4, \
                                 target_ycenter - target_ylen // 4, target_ycenter
            ref_target_slice[tx0: tx1, ty0: ty1] = arr[target_index, tx0: tx1, ty0: ty1]
            ref_candidates = np.nonzero(ref_target_slice)
            length = len(ref_candidates[0])
            if length == 0:
                return arr  # continue
            point_index = random.choice(list(range(length)))
            candidate = ref_candidates[0][point_index], ref_candidates[1][point_index]

            incre_x = min(piece_x1 - piece_x0 + 1, ref_target_slice.shape[0] - candidate[0])
            decre_y = min(piece_y1 - piece_y0 + 1, candidate[1])
            arr[target_index, candidate[0]: candidate[0] + incre_x,
            candidate[1] - decre_y: candidate[1]] \
                [piece_rot[piece_x0: piece_x0 + incre_x, piece_y1 - decre_y: piece_y1] == 1] = inter_value
        return arr


def convert_mat(_dir='/group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_train/'):

    save_dir = _dir + 'mat/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_paths = glob(_dir+'gt_label/*.nii.gz')
    image_paths = glob(_dir+'image/*.nii.gz')
    pred_paths = glob(_dir + 'pred/*.nii.gz')
    length = len(label_paths)

    for ith in range(length):
        label_path, image_path, pred_path = label_paths[ith], image_paths[ith], pred_paths[ith]
        label = nib.load(label_path).get_data()
        image = nib.load(image_path).get_data()
        pred  = nib.load(pred_path).get_data()
        sample = {'image': image, 'label': label, 'pred': pred}

        sample_name = label_path.split('/')[-1][:6] + '.mat'
        savemat(save_dir+sample_name, sample)







if __name__ == '__main__':

    # # convert .nii.gz to packed .mat
    # dirs = [
    #         '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_train/',
    #         '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_val/',
    #         '/group/lishl/weak_exp/output/0723_balance_r1_fgw1_auto_train/',
    #         '/group/lishl/weak_exp/output/0723_balance_r1_fgw1_auto_val/',
    #         '/group/lishl/weak_exp/output/0723_balance_r5_fgw1_auto_train/',
    #         '/group/lishl/weak_exp/output/0723_balance_r5_fgw1_auto_val/',
    #         ]
    # for _dir in dirs:
    #     convert_mat(_dir)

    # '''
    train_dir = '/group/lishl/weak_exp/output/1012_la_r1_01_train/'
    val_dir = '/group/lishl/weak_exp/output/1012_la_r1_01_val/'
    # train_dir = '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_train/'
    # val_dir = '/group/lishl/weak_exp/output/0722_balance_r3_fgw1_auto_val/'
    # train_dir = '/group/lishl/weak_exp/output/0723_balance_r1_fgw1_auto_train/'
    # val_dir = '/group/lishl/weak_exp/output/0723_balance_r1_fgw1_auto_val/'
    label_percent = 10
    anisotropy = False #if 'trachea' in train_dir else True
    save_dir = '/group/lishl/weak_exp/tmp/'
    import nibabel as nib

    batch_size = 1
    shuffle = False
    phase = 'train'     # 'val' #

    noise_params = {
        'extra_near_nums': '[0, 0]',  # extraneous unconnected regions: near branch center  [0, 2]    #Adjust1
        'extra_radius': '[2, 5]',  # radius [2, 5]
        'extra_slices': '[4, 10]',  # extraneous slices along z

        'dilate_nums': '[0, 0]',  # dilated regions near pred boundaries  [0: disable]  #Adjust3
        'dilate_iters': '[5, 11]',  # dilated iters(size) for each new region
        'extend_nums_head': '[0, 0]',  # elongation along trachea head
        'extend_nums_branch': '[0, 0]',  # elongation along trachea tail
        'close_iters': '0',  # closing operation iters for the whole region  [0: disable]

        'erode_nums':    '[0, 0]',
        'erode_iters':   '[3, 6]',
        'erode_slices':  '[5, 10]',
        'discard_nums':  '[1, 1]',
        'discard_slices': '[1, 3]',

        'close_iters': '[0, 0]',
        # 'extra_nums': '0',  # extraneous unconnected regions
        # 'extra_slices': '[10, 11]',  # extraneous slices along z
        # 'dilate_nums': '[1, 2]',  # dilated regions near pred boundaries
        # 'dilate_iters': '[3, 4]',  # dilated iters(size) for each new region
        # 'extend_nums_head': '[6, 10]',  # elongation along trachea head
        # 'extend_nums_branch': '[6, 10]',  # elongation along trachea tail
        # 'close_iters': '[3, 4]',  # closing operation iters for the whole region
    }

    dataset = AEDataset(train_dir, val_dir, batch_size, phase=phase, shuffle=shuffle,
                        noise_params=noise_params, use_pred_num=5, use_prob_top=True,
                        pseudo_label_lcc=True, anisotropy=anisotropy, label_percent=label_percent,
                        )
    print(len(dataset))
    from tqdm import tqdm
    for index in tqdm(range(300)):
        for ith, batch in enumerate(dataset):
            continue
            print(ith, batch.keys())

            print(batch['data'][0, 0].shape)
            print(batch['seg'][0, 0].shape)
            print(batch['pred'][0, 0].shape)
            import ipdb; ipdb.set_trace()

            nib.save(nib.Nifti1Image(batch['data'][0, 0], np.eye(4)), save_dir + '%02d_image.nii.gz' % ith)
            nib.save(nib.Nifti1Image(batch['seg'][0, 0], np.eye(4)), save_dir + '%02d_gt.nii.gz' % ith)
            nib.save(nib.Nifti1Image(batch['pred'][0, 0], np.eye(4)), save_dir + '%02d_pred.nii.gz' % ith)

    # train_dir = '/group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_train/'
    # val_dir = '/group/lishl/weak_exp/output/0723_trachea_balance_r3_fgw1_auto_ani_val/'
    # batch_size = 2
    #
    # # weak setting
    # extra_label_keys = ['pred']
    # use_weak = False
    # anisotropy = False  # for prostate
    #
    # # only for debug
    # # default_3D_augmentation_params['num_threads'] = 1
    # # default_3D_augmentation_params['num_cached_per_thread'] = 1
    #
    # # dataset
    # ds_train = AEDataset(train_dir, val_dir, batch_size, phase='train', shuffle=True, use_weak=use_weak)
    # ds_val = AEDataset(train_dir, val_dir, batch_size, phase='val', shuffle=False, use_weak=use_weak)
    # print(len(ds_train), len(ds_val))
    # patch_size = (next(ds_val))['data'].shape[-3:]
    # ipdb.set_trace()
    # '''
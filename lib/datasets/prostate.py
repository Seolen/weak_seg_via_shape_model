import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import cv2
import random
import pickle
import os.path as osp
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import h5py
import scipy.io
from lib.configs.parse_arg import opt, args


'''
# datadir
local_disk = True
if local_disk:
    data_dir = '/Volumes/Disk-2T/Shuailin_DATASET/Promise12/train_png/'
else:
    data_dir = '/group/lishl/weak_datasets/Promise12/train_png/'
'''


class Prostate_3D(Dataset):
    def __init__(self, phase):
        self.phase = phase
        self.data_dir = opt.data.data_dir
        self.data_folder = opt.data.data_folder
        self.case_names = self.load_case_names(phase)

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, item):
        case_name = self.case_names[item]
        case_dir = osp.join(self.data_dir, self.data_folder, case_name + '.mat')
        case_file = scipy.io.loadmat(case_name)
        image = torch.Tensor(case_file['image']).unsqueeze(dim=0)   # B, Z, H, W
        label = torch.Tensor(case_file['label']).unsqueeze(dim=0)

        return {
            'image': image,
            'gt': label,
            'weak': label
        }

    def load_case_names(self, phase):
        split_file = osp.join(self.data_dir, 'splits', '{}.txt'.format(phase))
        with open(split_file, 'r') as f:
            lines = f.readlines()
        case_names = []
        for line in lines:
            case_names.append(line.strip())
        return case_names


class Prostate_3D_delete(Dataset):
    def __init__(self, data_dir='/group/lishl/weak_datasets/Promise12/', phase='train', weak=False, tight_box=True,\
                 transforms=None, split_seed=0, intensity_threshold=(0, 1000)):
        super(Prostate_3D_delete, self).__init__()
        self.phase = phase
        self.weak = weak
        self.tight_box = tight_box
        self.transform = transforms

        # train/val split
        random.seed(split_seed)
        val_num = 10
        val_ids = random.sample(list(range(50)), val_num)
        train_ids = [p for p in list(range(50)) if p not in val_ids]
        assert len(val_ids) + len(train_ids) == 50
        print('Split Completed: val_ids %s' % str(val_ids))
        self.ids = train_ids if phase=='train' else val_ids

        # load images and masks
        if weak:
            self.prostate_path = osp.join(data_dir, 'prostate_w1.h5')
        else:
            self.prostate_path = osp.join(data_dir, 'prostate_f1.h5')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # load image and mask
        pid = self.ids[index]
        name = 'label'
        if self.phase == 'train' and self.weak and (not self.tight_box):
            name = 'weak'
        with h5py.File(self.prostate_path, 'r') as self.prostate:
            volume = self.prostate['%d/image' % (pid)][:]    # (C, H, W)
            label = self.prostate['%d/%s' % (pid, name)][:]

        # determine BG mask: only fg bbox if 'tight_box' else fg region
        if self.phase == 'train' and self.weak:
            if self.tight_box:
                # plot a tight box based on gt label
                label = self.build_box(label)
            else:
                # load scribble weak label
                label = self.build_scribble(label)
        '''
        # augment: flip, rotate
        if self.transform is None:
            tfs = RandomRotFlip(use_rotate=False, use_flip=False)
            sample = tfs({'image': volume, 'label': label})
            volume, label = sample['image'], sample['label']
        '''
        volume = torch.from_numpy(np.expand_dims(volume, axis=0)).float()
        label = torch.from_numpy(label).long()

        return {
            'image': volume,
            'label': label,
        }

    def build_box(self, gt_label, ignore_index=255):
        ''' for each slice, build tight box '''
        ''' gt_label: (S, W, H) '''
        weak_label = np.zeros_like(gt_label).astype(np.uint8)
        weak_label[:, :, :] = ignore_index

        for slice_idx in range(gt_label.shape[0]):
            if np.sum(gt_label[slice_idx]) == 0:
                weak_label[slice_idx, :, :] = 0
                continue
            nonzero = np.nonzero(gt_label[slice_idx])
            wmin, wmax, hmin, hmax = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
            weak_label[slice_idx, :wmin, :] = 0
            weak_label[slice_idx, wmax:, :] = 0
            weak_label[slice_idx, :, :hmin] = 0
            weak_label[slice_idx, :, hmax:] = 0

        return weak_label

    def build_scribble(self, weak_label, ignore_index=255):
        ''' weak fg determined, add extra bg '''
        label = weak_label
        bg_indexs = [10, 10, 10, 10]
        label[label == 0] = 255

        label[:, :bg_indexs[0], :] = 0
        label[:, :, :bg_indexs[2]] = 0
        label[:, -bg_indexs[1]:, :] = 0
        label[:, :, -bg_indexs[3]:] = 0

        return label




class Prostate_2D(Dataset):
    def __init__(self, data_dir='/group/lishl/weak_datasets/Promise12/train_png/', phase='train', weak=False, tight_box=True,\
                 box_prior=True, transforms=None,):
        super(Prostate_2D, self).__init__()
        self.phase = phase
        self.weak = weak
        self.tight_box = tight_box
        self.box_prior = box_prior  # provide box prior bounds
        self.transform = transforms

        # load images and masks
        _dir = osp.join(data_dir, phase + '_augment/')
        img_dir = osp.join(_dir, 'img/')
        gt_dir = osp.join(_dir, 'gt/')
        weak_dir = osp.join(_dir, 'box_tmp/')
        self.images = sorted(glob(img_dir+'*.png'))
        self.gts = sorted(glob(gt_dir+'*.png'))
        self.weaks = sorted(glob(weak_dir+'*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # load image and gt
        img = cv2.imread(self.images[index])
        img = np.transpose(img, (2, 0, 1))
        img = img[:1, :, :]     # one channel
        gt = cv2.imread(self.gts[index])[:, :, 0]
        weak = cv2.imread(self.weaks[index])[:, :, 0]

        # weak label reset to 255
        weak[weak == 1] = 255


        # transform
        img = img * 1.0 / 255

        # if self.weak and self.box_prior:
        #     box_prior = self.gen_box_prior(weak)

        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).long()
        weak = torch.from_numpy(weak).long()


        # if self.phase == 'train' and self.weak:
        #     sample = {
        #         'image': img,   'label': weak,  'ref_gt': gt,
        #     }
        # else:
        #     sample = {
        #         'image': img, 'label': gt,
        #     }
        sample = {
            'image': img, 'weak': weak, 'gt': gt,
        }

        # if self.box_prior:
        #     sample['box_prior'] = box_prior

        return sample

    def gen_box_prior(self, weak, d=5):
        weak = weak > 0     # assume only 1 FG box in weak labels
        shape = weak.shape
        masks_list, bounds_list = [], []

        if weak.sum() > 0:      # FG box exist
            # calculate FG box info
            nonzero = np.nonzero(weak)
            x1, y1, x2, y2 = nonzero[0].min(), nonzero[1].min(), nonzero[0].max(), nonzero[1].max()
            W, H = x2 - x1, y2 - y1

            # split M segments as box priors
            for i in range(W // d):
                mask = torch.zeros(shape, dtype=torch.float32)
                mask[x1 + i * d: x1 + (i + 1) * d, y1:y1 + H + 1] = 1
                masks_list.append(mask)
                bounds_list.append(d)

            if W % d:
                mask = torch.zeros(shape, dtype=torch.float32)
                mask[x1 + W - (W % d):x1 + W + 1, y1:y1 + H + 1] = 1
                masks_list.append(mask)
                bounds_list.append(W % d)

            for j in range(H // d):
                mask = torch.zeros(shape, dtype=torch.float32)
                mask[x1:x1 + W + 1, y1 + j * d:y1 + (j + 1) * d] = 1
                masks_list.append(mask)
                bounds_list.append(d)

            if H % d:
                mask = torch.zeros(shape, dtype=torch.float32)
                mask[x1:x1 + W + 1, y1 + H - (H % d):y1 + H + 1] = 1
                masks_list.append(mask)
                bounds_list.append(H % d)

        bounds = [torch.tensor(bounds_list, dtype=torch.float32)] if bounds_list else torch.zeros((0,), dtype=torch.float32)
        masks = torch.stack(masks_list) if masks_list else torch.zeros((0, *shape), dtype=torch.float32)

        out = (bounds, masks)
        # return {
        #     # 'masks': masks,
        #     'masks': [tuple(masks)],
        #     'bounds': [tuple(bounds)],
        # }
        return tuple(out)


class Prostate_2DE(Dataset):
    def __init__(self, data_dir='/group/lishl/weak_datasets/Promise12/train_slices_s1/', phase='train', weak=False, \
                 tight_box=True, transforms=None):
        super(Prostate_2DE, self).__init__()
        self.phase = phase
        self.weak = weak
        self.transform = transforms

        # load images and masks
        paths = ['images.npy', 'labels.npy', 'weaks.npy']
        self.images = np.load(data_dir + phase + '/' +paths[0])
        self.gts = np.load(data_dir + phase + '/' + paths[1])
        self.weaks = np.load(data_dir + phase + '/' + paths[2])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # load image and gt
        img, gt, weak = self.images[index], self.gts[index], self.weaks[index]
        img = np.expand_dims(img, axis=0)

        # transform
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).long()
        weak = torch.from_numpy(weak).long()
        # verify match
        # if gt.sum() > 0 and (weak==1).sum() == 0:
        #     print('Error: weak label have no valid FG.')

        if index % opt.label.divided_by != 0:
            # remove foreground label
            weak = weak + (weak == 1).long() * 254
            assert (weak == 1).sum() == 0

        # if self.phase == 'train' and self.weak:
        #     sample = {
        #         'image': img,   'label': weak,  'ref_gt': gt,
        #     }
        # else:
        #     sample = {
        #         'image': img, 'label': gt,
        #     }
        sample = {
            'image': img, 'weak': weak, 'gt': gt,
        }

        return sample

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import ipdb

    # data_dir = '/Users/seolen/Seolen-Project/_group/lishl/weak_datasets/Promise12/train_png/'
    data_dir = '/Volumes/Disk-2T/Shuailin_DATASET/Promise12/train_slices/'
    phase = 'train'
    dataset = Prostate_2DE(data_dir, phase, weak=True)
    for ith, sample in tqdm(enumerate(dataset)):
        img, label = sample['image'], sample['label']
        if label.sum() > 0:
            print(ith)
            ipdb.set_trace()
            gt = sample['ref_gt']
            box_prior = sample['box_prior']

        # if ith > 20:
        #     plt.imshow(img); plt.show()
        #     plt.imshow(label); plt.show()
        #     plt.imshow(gt); plt.show()
        #     ipdb.set_trace()

        # gt_arr = gt.numpy(); gt_30 = gt_arr[30]
        # plt.imshow(gt_30); plt.show()
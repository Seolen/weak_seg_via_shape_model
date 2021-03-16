import torch
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import cv2
import random
import os.path as osp
from lib.datasets.utils import RandomRotate, RandomFlip, RandomScale, RandomNoise


class Heart(Dataset):
    def __init__(self, data_dir='/group/lishl/weak_datasets/0108_SegTHOR/prior_crop/downsample', phase='train', weak=False,
                 transforms=None, split_seed=0):
        super(Heart, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.weak = weak
        self.transform = transforms

        # train/val split
        val_num = 8
        val_ids = list(range(40-val_num*(split_seed+1), 40-val_num*(split_seed)))
        train_ids = [p for p in list(range(40)) if p not in val_ids]
        assert len(val_ids) + len(train_ids) == 40
        print('Split Completed: val_ids %s' % str(val_ids))
        self.ids = train_ids if phase=='train' else val_ids

        # load images and masks
        if weak:
            pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # load image and mask
        pid = self.ids[index]
        image_path, label_path = '%02d.npy'%pid, '%02d_heart_label.npy' % pid
        image, label = np.load(osp.join(self.data_dir,image_path)), np.load(osp.join(self.data_dir, label_path))
        weak = np.load(osp.join(self.data_dir, '%02d_scribble_label.npy' % pid))

        # intensity window, augmentation
        window = [-986, 271]
        mean, std = 20.78, 180.50
        image[image < window[0]] = window[0]
        image[image > window[1]] = window[1]
        image = (image-mean) / std

        image = np.transpose(image, [2, 0, 1])
        label = np.transpose(label, [2, 0, 1])
        weak = np.transpose(weak, [2, 0, 1])

        # transform
        '''
        if self.phase=='train':
            sample = {'image': image, 'label': label, 'weak': weak}
            tf_rotate = RandomRotate(prob=0.2, rot90=False, degree=[-30, 30])
            # tf_flip = RandomFlip(prob=0.2, axes=[0, 1, 2])
            sample = tf_rotate(sample)
            # sample = tf_flip(sample)
            image, label, weak = sample['image'], sample['label'], sample['weak']
        '''

        '''
        # augment: flip, rotate
        if self.transform is None:
            tfs = RandomRotFlip(use_rotate=False, use_flip=False)
            sample = tfs({'image': volume, 'label': label})
            volume, label = sample['image'], sample['label']
        '''
        volume = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        label = torch.from_numpy(label).long()
        weak = torch.from_numpy(weak).long()

        return {
            'image': volume,
            'gt': label,
            'weak': weak,
        }


class HeartPatch(Dataset):
    # Patch based dataset
    def __init__(self, data_dir='/group/lishl/weak_datasets/0108_SegTHOR/prior_crop/patches', phase='train', weak=False,
                 transforms=None, split_seed=0):
        super(HeartPatch, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.weak = weak
        self.transform = transforms

        # train/val split
        val_num = 8
        val_ids = list(range(40-val_num*(split_seed+1), 40-val_num*(split_seed)))
        train_ids = [p for p in list(range(40)) if p not in val_ids]
        assert len(val_ids) + len(train_ids) == 40
        print('Split Completed: val_ids %s' % str(val_ids))
        self.ids = train_ids if phase=='train' else val_ids

        # load images and masks
        if weak:
            pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # load image and mask
        pid = self.ids[index]
        image_path, label_path = '%02d.npy'%pid, '%02d_heart_label.npy'%pid
        image, label = np.load(osp.join(self.data_dir,image_path)), np.load(osp.join(self.data_dir, label_path))
        image, label = self.pad_volume([image, label], output_size=[192, 180, 64])

        # intensity window, augmentation
        window = [-986, 271]
        mean, std = 20.78, 180.50
        image[image < window[0]] = window[0]
        image[image > window[1]] = window[1]
        image = (image-mean) / std

        image = np.transpose(image, [2, 0, 1])
        label = np.transpose(label, [2, 0, 1])

        '''
        # augment: flip, rotate
        if self.transform is None:
            tfs = RandomRotFlip(use_rotate=False, use_flip=False)
            sample = tfs({'image': volume, 'label': label})
            volume, label = sample['image'], sample['label']
        '''
        volume = torch.from_numpy(np.expand_dims(image, axis=0)).float()
        label = torch.from_numpy(label).long()

        return {
            'image': volume,
            'label': label,
        }

    def pad_volume(self, volumes, output_size=[192, 180, 64], pad_values=[-1000, 0]):
        image, label = volumes
        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2, 0)
            ph = max((output_size[1] - label.shape[1]) // 2, 0)
            pd = max((output_size[2] - label.shape[2]) // 2, 0)
            pw2 = output_size[0] - label.shape[0] - pw
            ph2 = output_size[1] - label.shape[1] - ph
            pd2 = output_size[2] - label.shape[2] - pd

            image = np.pad(image, [(pw, pw2), (ph, ph2), (pd, pd2)], mode='constant', constant_values=pad_values[0])
            label = np.pad(label, [(pw, pw2), (ph, ph2), (pd, pd2)], mode='constant', constant_values=pad_values[1])
            assert sorted(image.shape) == sorted(output_size)
            assert sorted(label.shape) == sorted(output_size)
        return image, label







if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import ipdb

    data_dir = '/Volumes/Disk-2T/Shuailin_DATASET/0108_SegTHOR/prior_crop/downsample/'
    phase = 'train'
    dataset = Heart(data_dir, phase)
    for ith, sample in enumerate(dataset):
        img, label = sample['image'], sample['gt']
        if img.shape[1] != 64:
            print(ith, img.shape)
        # ipdb.set_trace()
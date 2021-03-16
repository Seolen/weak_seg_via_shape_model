'''
load recist slices
'''

import numpy as np
import json
import pickle
import pydicom
import nibabel as nib
import cv2
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tfs #Resize, RandomHorizontalFlip, RandomRotation
# from data_util import grabcut
from tqdm import tqdm

from lib.datasets.data_util import grabcut, PadSquare, RecoverImage
# from data_util import grabcut, PadSquare, RecoverImage
from lib.opts import *

import ipdb
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt

# global data path
local_disk = False
# json_path = 'data.json'

if local_disk:
    recist_path = '../../../Recist/save/recist.pkl'
    mask_path = '../../../Recist/save/fully_mask.pkl'
    datadir = '/Volumes/Disk-2T/Shuailin_DATASET/Lymph_node/CT Lymph Nodes/'
else:   # server location
    _dir = '/group/lishl/weak_datasets/Lymph_node/'
    recist_path = _dir + 'process/recist.pkl'
    mask_path = _dir + 'process/fully_mask.pkl'
    datadir = _dir + 'CT Lymph Nodes/'

class LymphNode(Dataset):
    def __init__(self, transform=None, phase='train', out_size=(224, 224), range=(0, 120), pad=False, gt_mask=False):
        '''
        :param transform:
        :param phase:   'train'/'val'/'test'
        '''
        super(LymphNode, self).__init__()
        global opt
        opt = get_opt()
        self.out_size = out_size
        self.range = range

        # load json and recist files
        json_path = opt.data_dir + '/../network/lib/datasets/data.json'
        # json_path = 'data.json'
        self.dataset = json.load(open(json_path, 'r'))
        self.recist = pickle.load(open(recist_path, 'rb'))
        self.datadir = datadir
        self.data = self.dataset[phase]
        self.len = len(self.data.keys())
        self.phase = phase
        self.gt_mask = gt_mask  # whether use gt_mask at train phase
        if phase != 'train' or gt_mask:
            self.mask = pickle.load(open(mask_path, 'rb'))

        # methods
        self.window = Window()
        self.random_rotate = RandomRotate()
        self.random_flip = RandomFlip()
        if pad:
            self.pad = PadSquare()

    def __getitem__(self, index):
        '''
        :return:
            recist: shape (1, H, W)
        '''
        # load original image
        slice_info = self.data[str(index)]  # 'pid', 'label_id', 'roi', 'cslice'
        pid = int(slice_info['pid'])
        modality = 'ABD' if pid < 87 else 'MED'
        patient_id = pid if pid < 87 else pid - 86
        img_path = glob.glob(datadir + '%s_LYMPH_%03d/*/*/%06d.dcm'% (modality, patient_id, slice_info['cslice']))[0]
        img_ori = pydicom.dcmread(img_path).pixel_array

        roi = slice_info['roi']
        img_crop = img_ori[roi[0]: roi[1], roi[2]: roi[3]]
        img_crop = self.window(img_crop, range=self.range)

        # load label
        if self.phase == 'train' and not self.gt_mask:
            recist = self.recist[pid][int(slice_info['label_id'])]
            recist_crop = recist[roi[0]: roi[1], roi[2]: roi[3]]
            ## generate initial mask via grabcut
            img_c3 = self.convert_u8c3(img_crop)
            initial_mask = grabcut(img_c3, recist_crop, roi[:4])
        else:
            initial_mask = self.mask[pid][slice_info['label_id']][slice_info['cslice']]['mask_crop'].astype(np.uint8)
        assert img_crop.shape == initial_mask.shape

        # transformation a,b,c
        ### a. resize
        img = cv2.resize(img_crop, self.out_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(initial_mask, self.out_size, interpolation=cv2.INTER_NEAREST)

        if self.phase == 'train':
            ### b. random rotate/flip
            sample = {'image': img, 'label': label}
            sample = self.random_rotate(sample)
            sample = self.random_flip(sample)
            img, label = sample['image'], sample['label']

        ### c. normalize & toTensor
        eps = 0.00001
        img = 1.0 * (img - self.range[0]) / ((self.range[1] - self.range[0]) + eps)
        img = torch.from_numpy(np.expand_dims(img, axis=0).copy())
        label = torch.from_numpy(label.copy()).long()

        return img, label

    def __len__(self):
        return self.len

    def convert_u8c3(self, img):
        int_min, int_max = img.min(), img.max()
        img = (1.0 * (img - int_min) / (int_max - int_min) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    '''
    def window(self, img, range = (0, 120)):
        int_min, int_max = range
        img[img < int_min] = int_min
        img[img > int_max] = int_max
        return img


    def random_rotate(self, sample):
        # random rorate an image, which can be 2D or 3D, such as (C, H, W)
        img, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(img, k, axes=(-2, -1))
        label = np.rot90(label, k, axes=(-2, -1))
        return {'image': image, 'label': label}

    def random_flip(selfs, sample):
        # random flip an image, which can be 2D or 3D, such as (C, H, W)
        img, label = sample['image'], sample['label']
        k = -2 if (np.random.randint(0, 2) == 1) else -1
        image = np.flip(img, axis=k)
        label = np.flip(label, axis=k)
        return {'image': image, 'label': label}
    '''


#######################################################################################################

class LymphNode_FullMask(Dataset):
    def __init__(self, transform=None, phase='train', out_size=(224, 224), range=(0, 120)):
        '''
        :param transform:
        :param phase:   'train'/'val'/'test'
        '''
        super(LymphNode_FullMask, self).__init__()
        global opt
        opt = get_opt()
        self.out_size = out_size
        self.range = range

        # load json and mask files
        json_path = opt.data_dir + '/../network/lib/datasets/fully_data.json'
        # json_path = 'fully_data.json'
        self.dataset = json.load(open(json_path, 'r'))
        self.mask = pickle.load(open(mask_path, 'rb'))
        self.datadir = datadir
        self.data = self.dataset[phase]
        self.len = len(self.data.keys())
        self.phase = phase

        # methods
        self.window = Window()
        self.random_rotate = RandomRotate()
        self.random_flip = RandomFlip()


    def __getitem__(self, index):
        '''
        :return:
            mask: shape (1, H, W)
        '''
        # load mask, load original image
        slice_info = self.data[str(index)]  # 'pid', 'label_id', 'roi', 'slice'
        pid = int(slice_info['pid'])
        modality = 'ABD' if pid < 87 else 'MED'
        patient_id = pid if pid < 87 else pid - 86
        img_path = glob.glob(datadir + '%s_LYMPH_%03d/*/*/%06d.dcm'% (modality, patient_id, slice_info['slice']))[0]
        img_ori = pydicom.dcmread(img_path).pixel_array
        mask_crop = self.mask[pid][slice_info['label_id']][slice_info['slice']]['mask_crop'].astype(np.uint8)

        roi = slice_info['roi']
        img_crop = img_ori[roi[0]: roi[1], roi[2]: roi[3]]
        img_crop = self.window(img_crop, range=self.range)

        # transformation a,b,c
        ### a. resize
        img = cv2.resize(img_crop, self.out_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(mask_crop, self.out_size, interpolation=cv2.INTER_NEAREST)

        if self.phase == 'train':
            ### b. random rotate/flip
            sample = {'image': img, 'label': label}
            sample = self.random_rotate(sample)
            sample = self.random_flip(sample)
            img, label = sample['image'], sample['label']

        ### c. normalize & toTensor
        eps = 0.00001
        img = 1.0 * (img - self.range[0]) / ((self.range[1] - self.range[0]) + eps)
        img = torch.from_numpy(np.expand_dims(img, axis=0).copy())
        label = torch.from_numpy(label.copy()).long()

        return img, label

    def __len__(self):
        return self.len


#######################################################################################################

class Window(object):
    '''
    Window image intensity to a specific range
    img: array-like
    '''
    def __call__(self, img, range = (0, 120)):
        int_min, int_max = range
        img[img < int_min] = int_min
        img[img > int_max] = int_max
        return img

class RandomRotate(object):
    '''
    random rorate an image, which can be 2D or 3D, such as (C, H, W)
    '''
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(img, k, axes=(-2, -1))
        label = np.rot90(label, k, axes=(-2, -1))
        return {'image': image, 'label': label}

class RandomFlip(object):
    '''
    random flip an image, which can be 2D or 3D, such as (C, H, W)
    '''
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        k = -2 if (np.random.randint(0, 2) == 1) else -1
        image = np.flip(img, axis=k)
        label = np.flip(label, axis=k)
        return {'image': image, 'label': label}


#######################################################################################################

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    phase = 'test'
    # ds = LymphNode(phase=phase)
    ds = LymphNode_FullMask(phase=phase)
    print(len(ds))
    for ith, (img, label) in tqdm(enumerate(ds)):
        pass
        arr_img = img.numpy()[0]
        arr_mask = label.numpy()

        plt.imshow(arr_mask);
        plt.savefig('label_%03d.jpg' % ith)
        plt.imshow(arr_img, cmap='gray');
        plt.savefig('img_%03d.jpg' % ith)
        ipdb.set_trace()




'''
# Util
ipdb.set_trace()
plt.imshow(label); plt.savefig('label.jpg')
plt.imshow(im, cmap='gray'); plt.savefig('t.jpg')
plt.imshow(im_tensor[0].numpy(), cmap='gray'); plt.savefig('t2.jpg')


# plt.imshow(arr_img, cmap='gray');
        # plt.title('img');
        # plt.show()
        # plt.imshow(arr_mask);
        # plt.title('initial mask');
        # plt.show()
'''
import json
import numpy as np
import random
import cv2
from scipy import ndimage



class RandomRotate(object):
    """
    Random rotate a sample
    Two types:
        1. rotate with 90 degree as a unit
        2. rotate with random degree
    """
    def __init__(self, prob=0.2, rot90=True, degree=[-30, 30]):
        self.prob = prob
        self.rot90 = rot90
        self.degree = degree

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'weak' in sample.keys():
            weak = sample['weak']

        # random prob
        if random.random() < self.prob:
            if self.rot90:
                units = [2] if image.shape[-2]!=image.shape[-1] else [1,2,3]    # due to H doesn't match W
                k = random.choice(units)
                image = np.rot90(image, k, axes=(1, 2)).copy()      # along HW plane
                label = np.rot90(label, k, axes=(1, 2)).copy()
                if 'weak' in sample.keys():
                    weak = np.rot90(weak, k, axes=(1, 2)).copy()
            else:
                angle = random.uniform(self.degree[0], self.degree[1])
                image = self.rotate_xy(image, angle)
                label = self.rotate_xy(label, angle)
                if 'weak' in sample.keys():
                    weak = self.rotate_xy(weak, angle)
                    weak[weak>1] = 255  # filter interplated value

        output = {'image': image, 'label': label}
        if 'weak' in sample.keys():
            output['weak'] = weak

        return output

    def rotate_xy(self, volume, angle=0, scale=1.0):
        # volume: (S, H, W)
        S = volume.shape[0]
        output = np.zeros_like(volume, dtype=volume.dtype)
        for _slice in range(S):
            image = volume[_slice]
            (h, w) = image.shape[:2]
            center = (h/2, w/2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))
            output[_slice] = rotated
        assert sorted(volume.shape) == sorted(output.shape)
        return output



class RandomFlip(object):
    """
    Random flip a sample
    """
    def __init__(self, prob=0.2, axes=[0, 1, 2]):
        self.prob = prob
        self.axes = axes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'weak' in sample.keys():
            weak = sample['weak']

        # random prob
        if random.random() < self.prob:
            axis = random.choice(self.axes)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            if 'weak' in sample.keys():
                weak = np.flip(weak, axis=axis).copy()

        output = {'image': image, 'label': label}
        if 'weak' in sample.keys():
            output['weak'] = weak

        return output

class RandomScale(object):
    """
    Random scale a sample
    """
    def __init__(self, prob=0.2, scale_range=[0.8, 1.2]):
        self.prob = prob
        self.scale_range = scale_range

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'weak' in sample.keys():
            weak = sample['weak']

        # random prob
        if random.random() < self.prob:
            # resize
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            image_resize = ndimage.zoom(image, zoom=scale)
            label_resize = ndimage.zoom(label, zoom=scale)
            if 'weak' in sample.keys():
                weak_resize = ndimage.zoom(weak, zoom=scale)

            # crop or pad
            S, H, W = image.shape
            S2, H2, W2= image_resize.shape
            pads = [int(abs(S*(scale-1))//2), int(abs(H*(scale-1))//2), int(abs(W*(scale-1))//2)]
            if scale > 1.0:
                image = image_resize[pads[0]:pads[0]+S, pads[1]:pads[1]+H, pads[2]:pads[2]+W]
                label = label_resize[pads[0]:pads[0]+S, pads[1]:pads[1]+H, pads[2]:pads[2]+W]
                if 'weak' in sample.keys():
                    weak = weak_resize[pads[0]:pads[0] + S, pads[1]:pads[1] + H, pads[2]:pads[2] + W]

            elif scale < 1.0:
                image = np.zeros_like(image)
                image[pads[0]:pads[0]+S2, pads[1]:pads[1]+H2, pads[2]:pads[2]+W2] = image_resize
                label[pads[0]:pads[0] + S2, pads[1]:pads[1] + H2, pads[2]:pads[2] + W2] = label_resize
                if 'weak' in sample.keys():
                    weak[pads[0]:pads[0] + S2, pads[1]:pads[1] + H2, pads[2]:pads[2] + W2] = weak_resize

        output = {'image': image, 'label': label}
        if 'weak' in sample.keys():
            output['weak'] = weak

        return output


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


def stat_ratio():
    '''
    Statistics: w:h ratio histogram for three datasets
    '''
    index_file_path = 'data.json'
    index_file = json.load(open(index_file_path, 'r'))
    ratios = []

    phases = ['train', 'val', 'test']
    # phase = phases[0]
    for phase in phases:
        data = index_file[phase]
        for pid in data.keys():
            roi = data[pid]['roi']
            w = roi[1] - roi[0]
            h = roi[3] - roi[2]
            # ratio = 1.0*w/h if w>h else 1.0*h/w
            ratios.append(1.0*w/h)
        # plt.hist(ratios, bins=50); plt.title('%s ratio' % phase); plt.show()








if __name__ == '__main__':
    stat_ratio()
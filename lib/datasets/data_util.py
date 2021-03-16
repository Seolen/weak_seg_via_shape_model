'''
util: from recist to initial mask
'''

import numpy as np
import cv2
import math

def grabcut(img_crop, recist_crop, roi):
    ''' obtain initial mask via grabcut, img_crop and recist_crop as input '''
    # Parameter
    mask = np.zeros_like(img_crop[:, :, 0]).astype(np.uint8)
    nonzero = np.nonzero(recist_crop)
    try:
        x0, x1, y0, y1 = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
    except:
        import ipdb; ipdb.set_trace()

    rect = (x0, y0, x1-x0, y1-y0)

    # dilate recist FG
    ksize = 3
    kernel = np.ones((ksize, ksize), np.uint8)
    dilation = cv2.dilate(recist_crop, kernel, iterations=1)

    # set FG, BG and Pr_BG (1, 0, 2 respectively)
    w, h = roi[1]-roi[0], roi[3]-roi[2]
    inner_x0, inner_x1, inner_y0, inner_y1 = int(1.0*w/2-math.sqrt(2)/4*w), int(1.0*w/2+math.sqrt(2)/4*w),\
                                            int(1.0*h/2-math.sqrt(2)/4*h), int(1.0*h/2+math.sqrt(2)/4*h)
    mask[inner_x0: inner_x1, inner_y0: inner_y1] = 2
    mask[dilation > 0] = 1

    ## grabcut
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    iteration = 5
    # cv2.grabCut(img_crop, mask, rect, bgdModel, fgdModel, iteration, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(img_crop, mask, None, bgdModel, fgdModel, iteration, cv2.GC_INIT_WITH_MASK)
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2 = mask.copy()
    mask2[mask2 == 2] = 0; mask2[mask2 == 3] = 2   # PR_BG->BG (0); PR_FG-> (2)
    return mask2


class PadSquare(object):
    '''
    Pad array to unified ratio
    '''
    def __call__(self, sample, ratio=1.0, pad_value=0.0):
        img, label = sample['image'], sample['label']
        w, h = img.shape[0], img.shape[1]
        if w > h:
            # pad along h
            shape_pad = (w, int(w / ratio))
            img_pad = np.ones(shape_pad)
            label_pad = np.zeros(shape_pad)

        else:
            # pad along w
            shape_pad = (int(h/ratio), h)
            img_pad = np.ones(shape_pad)
            label_pad = np.zeros(shape_pad)

        img_pad[:w, :h] = img
        label_pad[:w, :h] = label
        info = {'shape_original': (w, h), 'shape_pad': shape_pad}
        sample = {
            'image': img_pad,
            'label': label_pad,
            'info': info,
        }
        return sample

class RecoverImage(object):
    '''
    Recover original mask from prediction
    mask: 2D array
    info: {'shape_original': (w, h), 'shape_pad': shape_pad}
    '''
    def __call__(self, mask, info):
        shape_ori = info['shape_original']
        shape_pad = info['shape_pad']
        label = cv2.resize(mask, self.shape_pad, interpolation=cv2.INTER_NEAREST)

        w, h = shape_ori
        label = label[:w, :h]
        return label

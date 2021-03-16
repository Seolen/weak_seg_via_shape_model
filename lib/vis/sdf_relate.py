import torch
import numpy as np
import nibabel as nib
import ipdb

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
def compute_sdf_np(arr, truncate_value=20):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(t) = 0; t in segmentation boundary
             +inf|t-b|; x in segmentation
             -inf|t-b|; x out of segmentation
    normalize sdf to [-1,1]
    """
    posmask = arr.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        posdis[posdis > truncate_value] = truncate_value
        negdis[negdis > truncate_value] = truncate_value
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        tsdf = (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis)) - \
              (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis))
        tsdf[boundary == 1] = 0

    return tsdf

def convert2sdf():
    caseid = 'Case09'
    source_path = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0825_pro_aelo_11_train/pred/%s_00.nii.gz' % caseid
    save_path = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0825_pro_aelo_11_train/tmp/%s.nii.gz' % caseid
    pred = nib.load(source_path).get_data()
    tsdf = compute_sdf_np(pred)
    nib.save(nib.Nifti1Image(tsdf, np.eye(4)), save_path)
    print('Done.')

if __name__ == '__main__':
    convert2sdf()

    # save_path = '/group/lishl/weak_exp/tmp/rank_pseudo.nii.gz'
    # nib.save(nib.Nifti1Image(pseudo.cpu().numpy().astype(np.uint8), np.eye(4)), save_path)
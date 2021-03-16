from scipy.stats import multivariate_normal
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np

import nibabel as nib

def eg_gaussian():
    # x1: prob; x2: distance
    mean, sigma3 = (0.5, 0), (0.5, 1.0)
    variance = (sigma3[0]/3, sigma3[1]/3)
    norm_func = multivariate_normal(mean, variance)

    inputs = [ (0.5, 0), (0.99, 0), (0.99, 0.99), (0.1, -0.99) ]
    for input in inputs:
        output = norm_func.pdf(input)
        print(input, '->', output)

def compute_sdf(arr, truncate_value=100):
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

    return tsdf, np.max(posdis), np.max(negdis)

def plot_gaussian():
    mean, sigma3 = (0.5, 0), (0.5, 1.0)
    variance = (sigma3[0] / 3, sigma3[1] / 3)
    norm_func = multivariate_normal(mean, variance)

    M = 500
    X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(-1, 1, M))
    d = np.dstack([X, Y])
    Z = norm_func.pdf(d).reshape(M, M)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(6, 4))
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def convert_sdf():

    max_distance = 20

    paths = [
        '/group/lishl/weak_exp/output/0825_trachea_aelo_31_train/pred/Case01_00.nii.gz',
        '/group/lishl/weak_exp/output/0825_pro_aelo_31_train/pred/Case00_00.nii.gz',
    ]
    save_dir = '/group/lishl/weak_exp/tmp/'

    for ith, path in enumerate(paths):
        pred = nib.load(path).get_data()
        tsdf, max_pos, max_neg = compute_sdf(pred, truncate_value=max_distance)
        print(max_pos, max_neg)

        save_path = save_dir + '%02d.nii.gz' % ith
        nib.save(nib.Nifti1Image(tsdf, np.eye(4)), save_path)


if __name__ == '__main__':
    # eg_gaussian()
    convert_sdf()

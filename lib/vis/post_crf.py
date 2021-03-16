import nibabel as nib
import denseCRF3D
import numpy as np
from glob import glob
import os
import ipdb
import datetime

def densecrf3d(image_path, pred_path, save_path):
    I1NII = nib.load(image_path)
    I1 = I1NII.get_data()
    PNII = nib.load(pred_path)
    P = PNII.get_data()

    # convert input to intenstiy range of [0, 255]
    I = np.asarray([I1], np.float32)
    I = np.transpose(I, [1, 2, 3, 0])
    I = I / I.max() * 255
    I = np.asarray(I, np.uint8)

    # probability map for each class
    P = 0.5 + (P - 0.5) * 1.0
    P = np.asarray([1.0 - P, P], np.float32)
    P = np.transpose(P, [1, 2, 3, 0])

    dense_crf_param = {}
    dense_crf_param['MaxIterations'] = 2.0
    dense_crf_param['PosW'] = 2.0
    dense_crf_param['PosRStd'] = 5
    dense_crf_param['PosCStd'] = 5
    dense_crf_param['PosZStd'] = 5
    dense_crf_param['BilateralW'] = 3.0
    dense_crf_param['BilateralRStd'] = 5.0
    dense_crf_param['BilateralCStd'] = 5.0
    dense_crf_param['BilateralZStd'] = 5.0
    dense_crf_param['ModalityNum'] = 1
    dense_crf_param['BilateralModsStds'] = (5.0,)

    lab = denseCRF3D.densecrf3d(I, P, dense_crf_param)
    # lab = denseCRF3D.densecrf3d(I[0:1], P[0:1], dense_crf_param)  for test one slice time.

    labNii = nib.Nifti1Image(lab, np.eye(4))
    nib.save(labNii, save_path)

def demo_crf(_dir):
    # _dir = '/group/lishl/weak_exp/output/1012_tra_r5_01_train/'
    # image_path = _dir + 'image/Case05.nii.gz'
    # pred_path = _dir + 'heatmap/Case05.nii.gz'
    # save_path = _dir + 'crf_' + 'Case05.nii.gz'

    paths = {'image': sorted(glob(_dir + 'image/*.nii.gz')),
             'prob': sorted(glob(_dir + 'heatmap/*.nii.gz')),}
    savedir = _dir + 'postcrf/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for image_path, pred_path in zip(paths['image'], paths['prob']):
        case_name = image_path.split('/')[-1]
        save_path = savedir + case_name
        densecrf3d(image_path, pred_path, save_path)

def grabcut(image_path, pred_path, weak_path, save_path):
    import imcut.pycut as pspc
    image = nib.load(image_path).get_data()
    weak = nib.load(weak_path).get_data()
    pred = nib.load(pred_path).get_data()

    seeds = weak.copy()
    seeds[pred == 1] = 3    # uncertain FG
    seeds[pred == 0] = 4    # uncertain BG
    seeds[weak == 1] = 1    # fg
    seeds[weak == 0] = 2    # background

    # current = datetime.datetime.now()
    # run
    igc = pspc.ImageGraphCut(image, voxelsize=[1, 1, 1])
    igc.set_seeds(seeds)
    igc.run()
    # print('Cost', datetime.datetime.now() - current)
    # save results
    pred = igc.segmentation
    pred = (1 - pred).astype(np.uint8)

    labNii = nib.Nifti1Image(pred, np.eye(4))
    nib.save(labNii, save_path)

def demo_grabcut(_dir, weak_dir=None):
    paths = {'image': sorted(glob(_dir + 'image/*.nii.gz')),
             'pred': sorted(glob(_dir + 'pred/*.nii.gz')), }
    if weak_dir is not None:
        case_names = [_path.split('/')[-1].split('.')[0] for _path in paths['image']]
        paths['weak'] = [weak_dir + case_name + '_weak_label.nii.gz' for case_name in case_names]
    savedir = _dir + 'postgrab/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for ith in range(len(paths['image'])):
        image_path, pred_path, weak_path = paths['image'][ith], paths['pred'][ith], paths['weak'][ith]
        case_name = image_path.split('/')[-1]
        save_path = savedir + case_name
        grabcut(image_path, pred_path, weak_path, save_path)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default='', help="")
    args = vars(ap.parse_args())

    Params = {
        'stage':    'densecrf'   # {'densecrf', 'grabcut', }
    }

    dirprefix = '/group/lishl/weak_exp/output/'
    if Params['stage'] == 'densecrf':
        dirs = [
            # '1016_trar5_emita_01_val',
            # '1016_trar3_emita_03_val',
            # '1016_trar1_emita_23_val',
            # '1013_tra_aelo_53_val',
            # '1013_tra_aelo_38_val',
            # '1013_tra_aelo_13_val',
            '1012_tra_r1_01_val',
            '1012_tra_r3_01_val',
            '1012_tra_r5_01_val',
        ]

        # argparse replace
        if args['id'] != '':
            dirs = [args['id'] + '_val']

        for _dir in dirs:
            _dir = dirprefix + _dir + '/'
            demo_crf(_dir)

    elif Params['stage'] == 'grabcut':
        dirs = [
            # '1016_trar5_emita_01_val',
            '1016_trar3_emita_03_val',
            '1016_trar1_emita_23_val',
        ]
        weak_dirs = [
            # '/group/lishl/weak_datasets/0108_SegTHOR/1011_trachea_train_weak_percent_0.5_random/expand_nii/',
            '/group/lishl/weak_datasets/0108_SegTHOR/1011_trachea_train_weak_percent_0.3_random/expand_nii/',
            '/group/lishl/weak_datasets/0108_SegTHOR/1011_trachea_train_weak_percent_0.1_random/expand_nii/',
        ]
        for ith, _dir in enumerate(dirs):
            _dir = dirprefix + _dir + '/'
            weak_dir = weak_dirs[ith]
            demo_grabcut(_dir, weak_dir)
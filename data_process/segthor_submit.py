import os
import numpy as np
import nibabel as nib
import shutil
import skimage
from skimage import measure
import ipdb
import argparse

def reorganize_train_labels():
    root = '/group/lishl/weak_datasets/0108_SegTHOR/train/'
    save_dir = '/group/lishl/weak_datasets/0108_SegTHOR/train_labels/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for subdir in sorted(os.listdir(root)):
        if 'Patient' in subdir:
            src = root + subdir + '/GT.nii.gz'
            case_id = subdir[-2:]
            dst = save_dir + 'Case' + case_id + '.nii.gz'
            shutil.copy(src, dst)
    print('Labels re-organized in ', save_dir)

def rename_for_summit(label_dir, save_dir):
    def getLargestCC(segmentation):
        labels = measure.label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC.astype(np.uint8)

    for path in sorted(os.listdir(label_dir)):
        if '.nii.gz' in path:
            case_id = path[4:6]
            nii = nib.load(label_dir + path)
            label = nii.get_data()

            if Params['use_lcc']:
                label = getLargestCC(label)
            label[label > 0] = 3    # segthor id


            if Params['post_shift_y'] != 0:
                label_new = np.zeros_like(label)
                shift_y = Params['post_shift_y']
                if shift_y < 0:
                    label_new[:, :shift_y, :] = label[:, -shift_y:, :]
                else:
                    label_new[:, shift_y:, :] = label[:, :-shift_y, :]
                label = label_new
            if Params['post_shift_x'] != 0:
                label_new = np.zeros_like(label)
                shift_x = Params['post_shift_x']
                if shift_x < 0:
                    label_new[:shift_x, :, :] = label[-shift_x:, :, :]
                else:
                    label_new[shift_x:, :, :] = label[:-shift_x, :, :]
                label = label_new

            if Params['add_other_label']:
                # add other labels: 1, 2, 4
                label[1, 0, 0] = 1
                label[2, 0, 0] = 2
                label[4, 0, 0] = 4

            save_name = 'Patient_' + case_id + '.nii'
            nib.save(nib.Nifti1Image(label, nii.affine), save_dir+save_name)

def obtain_result():

    label_dir = dirprefix + exp_id + '/medimg_nii/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    rename_for_summit(label_dir, result_dir)
    shutil.make_archive(dirprefix+exp_id+'/%s' % (exp_id), 'zip', result_dir)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default='', help="")
    args = vars(ap.parse_args())

    Params= {
        'exp_id': '1016_trachea_fully',
        # 1016_trar1_emitw_24, 1016_trar3_emitw_04, 1016_trar5_emitw_24, 1016_trar10_emitw_02
        'add_other_label': True,

        'use_lcc': True,   # True
        'post_shift_y': 0,  # default 0
        'post_shift_x': 0,  # default 0
    }
    if args['exp_id'] != '':
        Params['exp_id'] = args['exp_id']

    exp_id = Params['exp_id']
    dirprefix = '/group/lishl/weak_datasets/0108_SegTHOR/test_pred/'
    result_dir = dirprefix + exp_id + '/result/'
    print('Processing, ', dirprefix + exp_id)
    print('use_lcc', Params['use_lcc'])
    obtain_result()
    # shutil.make_archive(result_dir[:-1], 'zip', result_dir)
    print('Saved result.zip in ', result_dir)
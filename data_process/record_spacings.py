from scipy.io import loadmat
import json
from process_pipeline2 import NpEncoder
import ipdb
import glob
import nibabel as nib
import SimpleITK as sitk

def record_orginal(general_dir, log_path):
    data = loadmat(general_dir + 'load.mat')
    log = {}

    for case_name in data.keys():
        if 'Case' not in case_name:
            continue
        log[case_name] = {
            'spacing': data[case_name]['spacing'][0, 0][0],     # default shape (1,3), load as shape [3]
            'shape': data[case_name]['label'][0, 0].shape
        }
    with open(log_path, 'w') as f:
        json.dump(log, f, cls=NpEncoder)
    print('Record spacings and shapes done! See %s', log_path)

def record_orginal_test(general_dir, log_path):
    image_paths = glob.glob(general_dir + '*.nii.gz')
    image_paths = [image_path for image_path in image_paths if '_label' not in image_path]
    log = {}

    for path in sorted(image_paths):
        if Params['data_name'] == 'LA':
            case_name = path.split('/')[-1][:6]
        elif Params['data_name'] == 'Segthor':
            case_id = path.split('/')[-1][8:10]; case_name = 'Case'+case_id
        nii_image = sitk.ReadImage(path)
        spacing = nii_image.GetSpacing()
        image = nib.load(path).get_data()
        shape = image.shape
        if Params['depth_transpose']:
            spacing = [spacing[2], spacing[0], spacing[1]]
            shape = [shape[2], shape[0], shape[1]]
        log[case_name] = {
            'spacing': spacing,     # default shape (1,3), load as shape [3]
            'shape': shape
        }
    with open(log_path, 'w') as f:
        json.dump(log, f, cls=NpEncoder)
    print('Record spacings and shapes done! See %s', log_path)

if __name__ == '__main__':
    '''
    Note: Log write original spacing and shape of each case, for recovering labels.
    '''

    Params = {
        'data_name':    'Segthor',      # {'Promise', 'Segthor', 'LA'}
        'depth_transpose': True,    # LA, Segthor need depth_transpose

        'log_name':     'original.json',# contain shapes and original spacings of each case
    }

    General_dirs = {
        # 'Promise': '/group/lishl/weak_datasets/Promise12/1009_train_weak_random/',
        'Segthor': '/group/lishl/weak_datasets/0108_SegTHOR/test/',
        'LA':      '/group/lishl/weak_datasets/LA_dataset/test/',

    }
    general_dir = General_dirs[Params['data_name']]
    log_path = general_dir + Params['log_name']

    # record_orginal(general_dir, log_path)
    record_orginal_test(general_dir, log_path)
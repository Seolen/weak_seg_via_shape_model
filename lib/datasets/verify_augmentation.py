import ipdb
import os
import numpy as np
import nibabel as nib
from batch_dataset import BGDataset #lib.datasets.
from batch_augmentation import get_moreDA_augmentation
data_dir = '/group/lishl/weak_datasets/0108_SegTHOR/processed_train/crop_mat/'
batch_size = 1
shuffle = False

def makedir(dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)


# dataset
ds_train = BGDataset(data_dir, batch_size, phase='train', shuffle=shuffle)
ds_val = BGDataset(data_dir, batch_size, phase='train', shuffle=False)  # the same for comparing augmentation
print(len(ds_train), len(ds_val))
print(next(ds_train)['data'].shape)

patch_size = [112, 240, 272]    # or (next(ds_train))['data'].shape[-3:]
# dataloader: default augmentation
train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size)
val_batch = next(val_loader)

# dataloader: add deep supervision branch
net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
    np.vstack(net_num_pool_op_kernel_sizes), axis=0))[:-1]
print(deep_supervision_scales)
train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size,\
                                                   deep_supervision_scales=deep_supervision_scales)
train_batch = next(next(train_loader))
val_batch = next(next(val_loader))
print(train_batch.keys(), train_batch['data'].shape, len(train_batch['target']))



# save || val_batch['data'][0], val_batch['target'][0]


verify_dir = '/group/lishl/weak_datasets/0108_SegTHOR/processed_train/verify'
batch_id = 'patient' + '_%02d' % 0
batch_dir = os.path.join(verify_dir, batch_id)
makedir([verify_dir, batch_dir])

data = {
    'data': val_batch['data'].numpy()[0, 0, ...],
    'target': val_batch['target'][0].numpy()[0, 0 ,...],
    'target_d1': val_batch['target'][1].numpy()[0, 0, ...],
    'target_d2': val_batch['target'][2].numpy()[0, 0, ...],
    'target_d3': val_batch['target'][3].numpy()[0, 0, ...],
}
data_aug = {
    'data': train_batch['data'].numpy()[0, 0, ...],
    'target': train_batch['target'][0].numpy()[0, 0 ,...],
    'target_d1': train_batch['target'][1].numpy()[0, 0, ...],
    'target_d2': train_batch['target'][2].numpy()[0, 0, ...],
    'target_d3': train_batch['target'][3].numpy()[0, 0, ...],
}


ipdb.set_trace()

for key, value in data.items():
    save_path = os.path.join(batch_dir, key+'.nii.gz')
    img = nib.Nifti1Image(value, np.eye(4))
    nib.save(img, save_path)

for key, value in data_aug.items():
    save_path = os.path.join(batch_dir, key+'_aug.nii.gz')
    img = nib.Nifti1Image(value, np.eye(4))
    nib.save(img, save_path)


from copy import deepcopy

import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, \
    GammaTransform, Compose #, MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from lib.datasets.transform.util_downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2, \
                                            Convert2DTo3DTransform, Convert3DTo2DTransform      #lib.datasets.
from lib.datasets.transform.spatial_transforms import SpatialTransform, MirrorTransform, RemoveLabelTransform#lib.datasets.

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,  #our dataset need not to choose channel

    # SpatialTransform Params

    "do_elastic": False,    #True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.7, 1.4),  #(0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),    #(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),    #(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),    #(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,            # True if anisotropic data
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 4, #12, shuailin modify
    "num_cached_per_thread": 2,     #1
}

def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True,
                            anisotropy=False,
                            extra_label_keys=None,
                            val_mode=False,
                            use_conf=False,
                            ):
    '''
    Work as Dataloader with augmentation
    :return: train_loader, val_loader
        for each iterator, return {'data': (B, D, H, W), 'target': (B, D, H, W)}
    '''

    if not val_mode:
        assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

        tr_transforms = []
        if params.get("selected_data_channels") is not None:
            tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        # anistropic setting
        if anisotropy or params.get("dummy_2D"):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform(extra_label_keys=extra_label_keys))
            patch_size = patch_size[1:]     # 2D patch size

            print('Using dummy2d data augmentation')
            params["elastic_deform_alpha"] = (0., 200.)
            params["elastic_deform_sigma"] = (9., 13.)
            params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
            params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

        else:
            ignore_axes = None



        # 1. Spatial Transform: rotation, scaling
        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg,
            order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis"),
            extra_label_keys=extra_label_keys
        ))

        if anisotropy or params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform(extra_label_keys=extra_label_keys))

        # 2. Noise Augmentation: gaussian noise, gaussian blur
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))

        # 3. Color Augmentation: brightness, constrast, low resolution, gamma_transform
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        if params.get("do_additive_brightness"):
            tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                     params.get("additive_brightness_sigma"),
                                                     True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                     p_per_channel=params.get("additive_brightness_p_per_channel")))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=0.1))  # inverted gamma
        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))

        # 4. Mirror Transform
        if params.get("do_mirror") or params.get("mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes"), extra_label_keys=extra_label_keys))

        # if params.get("mask_was_used_for_normalization") is not None:
        #     mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        #     tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0, extra_label_keys=extra_label_keys))
        tr_transforms.append(RenameTransform('data', 'image', True))
        tr_transforms.append(RenameTransform('seg', 'gt', True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'gt', 'gt', classes))
            else:
                tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='gt',
                                         output_key='gt', extra_label_keys=extra_label_keys))
        toTensorKeys = ['image', 'gt'] + extra_label_keys if extra_label_keys is not None else ['image', 'gt']
        tr_transforms.append(NumpyToTensor(toTensorKeys, 'float'))
        tr_transforms = Compose(tr_transforms)

        if seeds_train is not None:
            seeds_train = [seeds_train] * params.get('num_threads')
        if use_conf:
            num_threads = 1
            num_cached_per_thread = 1
        else:
            num_threads, num_cached_per_thread = params.get('num_threads'), params.get("num_cached_per_thread")
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_threads,
                                                      num_cached_per_thread,
                                                      seeds=seeds_train, pin_memory=pin_memory)

        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0, extra_label_keys=extra_label_keys))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
        val_transforms.append(RenameTransform('data', 'image', True))
        val_transforms.append(RenameTransform('seg', 'gt', True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'gt', 'gt', classes))
            else:
                val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='gt',
                                                                   output_key='gt', extra_label_keys=extra_label_keys))

        val_transforms.append(NumpyToTensor(toTensorKeys, 'float'))
        val_transforms = Compose(val_transforms)

        # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
        #                                             params.get("num_cached_per_thread"),
        #                                             seeds=seeds_val, pin_memory=pin_memory)
        if seeds_val is not None:
            seeds_val = [seeds_val] * 1
        # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, 1,
        #                                             params.get("num_cached_per_thread"),
        #                                             seeds=seeds_val, pin_memory=False)
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    else:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0, extra_label_keys=extra_label_keys))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))
        val_transforms.append(RenameTransform('data', 'image', True))
        val_transforms.append(RenameTransform('seg', 'gt', True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'gt', 'gt', classes))
            else:
                val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='gt',
                                                                   output_key='gt', extra_label_keys=extra_label_keys))

        toTensorKeys = ['image', 'gt'] + extra_label_keys if extra_label_keys is not None else ['image', 'gt']
        val_transforms.append(NumpyToTensor(toTensorKeys, 'float'))
        val_transforms = Compose(val_transforms)
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
        if dataloader_train is not None:
            batchgenerator_train = SingleThreadedAugmenter(dataloader_train, val_transforms)
        else:
            batchgenerator_train = None

    return batchgenerator_train, batchgenerator_val






if __name__ == '__main__':
    '''
    Pipeline:
        1. define ds_train, ds_val      [BGDataset(), params: (data_dir, batch_size, phase, shuffle ...)]
            return shape (B, C, D, W, H) for 3D volume, where C=1
        2. define patch_size            [D, W, H]
        3. define dataloader with augmentation      [MultiThreadedAugmenter(), params: (ds, tranforms, ...)]
        4. iteratively obtain 'data' and 'target' from dataloader
            return shape (B, C, D, W, H) for 'data' and 'target'   
            
        More. add deep supervision branch - return multiple targets in different scales
        Egample:
        - deep_supervision_scales = [[1, 1, 1], [1.0, 0.5, 0.5], [0.5, 0.25, 0.25]]
        - output
            - 'data': (B, C, D, W, H)
            - 'target': [(B, C, D*1, W*1, H*1), (B, C, D*1, W*0.5, H*0.5), (B, C, D*5, W*0.25, H*0.25)]
    '''


    import ipdb
    import torch
    import nibabel as nib
    from batch_dataset import BGDataset #lib.datasets.
    # data_dir = '/group/lishl/weak_datasets/0108_SegTHOR/processed_train/normalize_mat/'
    data_dir = '/group/lishl/weak_datasets/Promise12/processed_train_weak_stride_1/normalize_mat/'
    batch_size = 2

    # weak setting
    extra_label_keys = ['weak_label']
    use_weak = True
    use_duplicate = False
    anisotropy = True   # for prostate

    # only for debug
    # default_3D_augmentation_params['num_threads'] = 1
    # default_3D_augmentation_params['num_cached_per_thread'] = 1

    # dataset
    ds_train = BGDataset(data_dir, batch_size, phase='train', shuffle=True, use_weak=use_weak, use_duplicate=use_duplicate)
    ds_val = BGDataset(data_dir, batch_size, phase='val', shuffle=False, use_weak=use_weak, use_duplicate=use_duplicate)
    print(len(ds_train), len(ds_val))
    patch_size = (next(ds_val))['data'].shape[-3:]


    # dataloader: default augmentation
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
        np.vstack(net_num_pool_op_kernel_sizes), axis=0))[:-1]
    print(deep_supervision_scales)
    train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size,
                                   deep_supervision_scales=deep_supervision_scales, extra_label_keys=extra_label_keys,
                                    anisotropy=anisotropy)
    ipdb.set_trace()
    train_batch = next(train_loader)
    print(torch.unique(train_batch['weak_label']))
    val_batch = next(val_loader); print(torch.unique(val_batch['weak_label']))
    print(sum(1 for _ in train_loader)); print(sum(1 for _ in val_loader))


    # save and visualize
    arrs = [arr[0, 0].numpy() for arr in train_batch['gt']]
    [nib.save(nib.Nifti1Image(arr, np.eye(4)), 'train_gt_%02d.nii.gz'%ith) for (ith, arr) in enumerate(arrs)]
    arrs = [arr[0, 0].numpy() for arr in train_batch['weak_label']]
    [nib.save(nib.Nifti1Image(arr, np.eye(4)), 'train_weak_%02d.nii.gz' % ith) for (ith, arr) in enumerate(arrs)]



    train_batch = next(train_loader)
    print(train_batch.keys())           # dict_keys(['data', 'target']), each with torch.Size([2, 1, 112, 240, 272])
    val_batch = next(val_loader)






    # dataloader: add deep supervision branch
    ipdb.set_trace()
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
        np.vstack(net_num_pool_op_kernel_sizes), axis=0))[:-1]
    print(deep_supervision_scales)
    train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size,\
                                                       deep_supervision_scales=deep_supervision_scales)
    train_batch = next(train_loader)
    print(train_batch.keys(), train_batch['data'].shape, len(train_batch['target']))
    print(train_batch['target'][0].shape, train_batch['target'][1].shape, train_batch['target'][2].shape)
    val_batch = next(val_loader)
    # '''


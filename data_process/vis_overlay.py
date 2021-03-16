'''
INPUT root_dir
OUTPUT overlay_dir where pred, gt overlayed on images
'''

import os
import cv2
import numpy as np
import nibabel as nib

# root_dir = '/Users/seolen/Seolen-Project/_root/PycharmProjects/2002_semi_segmentation/weak_semi_medical_seg/output/0607_vis01_train/'
root_dir = '/Users/seolen/Seolen-Project/_group/lishl/weak_exp/output/0616_vis_heart_train/'
dataset_2d = False   # 2d or 3d dataset
img_dir = os.path.join(root_dir, 'image/')
pred_dir = os.path.join(root_dir, 'pred/')
gt_dir = os.path.join(root_dir, 'gt_label/')
save_dir = os.path.join(root_dir, 'overlay/')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_paths = sorted(os.listdir(img_dir))
pred_paths = sorted(os.listdir(pred_dir))
gt_paths = sorted(os.listdir(gt_dir))


if dataset_2d:
    for ith in range(len(img_paths)):
        img = cv2.imread(os.path.join(img_dir, img_paths[ith]))
        label = cv2.imread(os.path.join(pred_dir, pred_paths[ith]))
        label[:, :, :-1] = 0  # pred: red color
        gt = cv2.imread(os.path.join(gt_dir, gt_paths[ith]))
        gt[:, :, 0] = gt[:, :, 2] = 0  # gt: green color (intersection will be yellow)

        label = label + gt
        label[label == 0] = img[label == 0]

        combine = cv2.addWeighted(img, 0.7, label, 0.3, 0)
        cv2.imwrite(os.path.join(save_dir, img_paths[ith]), combine)
else:
    for filepath in sorted(os.listdir(img_dir)):
        image_path, gt_path, pred_path = img_dir+filepath, gt_dir+filepath, pred_dir+filepath
        image, gt, pred = nib.load(image_path).get_data(), nib.load(gt_path).get_data(), nib.load(pred_path).get_data()
        for slice_index in range(image.shape[0]):
            png_image, png_gt, png_pred = image[slice_index], gt[slice_index], pred[slice_index]
            save_path = save_dir+'%s_%02d.png'%(filepath.split('.')[0], slice_index)
            output = (png_image - png_image.min())/((png_image.max()) - png_image.min())*255
            # overlay: img+gt
            out_img = np.repeat(np.expand_dims(output, axis=2), 3, axis=2).astype(np.uint8)
            out_gt = np.zeros_like(out_img, dtype=np.uint8)
            out_gt[:, :, 1] = png_gt*255    # green
            overlay = cv2.addWeighted(out_img, 0.7, out_gt, 0.3, 0)
            # overlay_2: img+gt+pred
            out_pred = np.zeros_like(out_img, dtype=np.uint8)
            out_pred[:, :, -1] = png_pred * 255  # red
            label = out_gt+out_pred
            overlay_2 = cv2.addWeighted(out_img, 0.7, label, 0.3, 0)

            cv2.imwrite(save_path, overlay_2)

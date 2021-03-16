import numpy as np
import torch
import torch.nn as nn
import math
import copy
import os
import ipdb
from lib.configs.parse_arg import opt, args


def get_pseudo_mask(features, preds, targets, distance_matrix, inputs=None, i_batch=None):
    '''
    Get pseudo mask
    :param features: (B, C, H, W)
    :param preds: (B, H, W)
    :param targets: (B, H, W)   fg: 1, bg: 0, unlabeled: 255
    :return: soft_label: (B, 2, H, W)
    '''

    # params
    lambda_1 = opt.similarity.lambda_1 # 1.
    top_thresh = opt.similarity.top_thresh # 0.1
    label_channel = opt.similarity.label_channel
    B, C, H, W = features.shape
    ## print(features.shape, preds.shape, targets.shape)

    # feature similarity
    distance_matrix = distance_matrix[:B]   # in case of batch_size=1
    down_scale = preds.shape[-1] // W
    features = features / features.norm(dim=1).unsqueeze(dim=1).expand(B, C, H, W)      # normalize feature
    features_preds = features.flatten(start_dim=2).permute(0, 2, 1)                     # reshape to [B, H*W, C]
    features_targets = features.flatten(start_dim=2)                                    # reshape to [B, C, H*W]
    similarity_matrix = features_preds.bmm(features_targets)                            # dense similarity:[B, H*W, H*W]
    similarity_matrix = (similarity_matrix + 1) / 2

    # weighted feature similarity: current not invoke
    weight_matrix = similarity_matrix * 0.
    if 'simi' in opt.similarity.component:
        weight_matrix = similarity_matrix
    if 'dist' in opt.similarity.component:
        weight_matrix = weight_matrix / distance_matrix * lambda_1
    weight_max, _ = weight_matrix.max(dim=-1)           # 这里会是1，自己与自己的相似度为1
    weight_matrix /= weight_max.unsqueeze(dim=-1)

    # if i_batch == 5:
        # ipdb.set_trace()

    # downsample preds
    preds_reserved = preds.clone()
    preds = replace_labeled_pixels(preds, targets, label='fg_bg')
    pool = nn.AvgPool2d(kernel_size=down_scale, stride=down_scale, padding=0)
    preds = pool(preds.unsqueeze(dim=1)).squeeze(dim=1)
    preds = preds.flatten(start_dim=1).unsqueeze(dim=1).expand(B, H * W, H * W)

    # calculate soft labels
    fg_num = int((preds[0, 0] > 0.5).float().sum().item())
    if fg_num > 0:
        fg_weight_matrix = weight_matrix * (preds > 0.5).float()
        fg_bound = torch.topk(fg_weight_matrix, fg_num, dim=-1)[0][:, :, -1:]
        fg_mask = (fg_weight_matrix >= fg_bound).float()
        fg_weight = fg_mask * weight_matrix
        # fg_max, _ = fg_weight.max(dim=-1)
        # fg_weight /= fg_max.unsqueeze(dim=-1)
        fg_weight_avg = fg_weight / fg_mask.sum(dim=-1).unsqueeze(dim=-1)
    else:
        fg_mask = weight_matrix.clone() * 0
        fg_weight = weight_matrix.clone() * 0.
        fg_weight_avg = fg_weight

    bg_num = int((preds[0, 0] < 0.5).float().sum().item() * top_thresh)
    bg_num = min(bg_num, 15)
    if bg_num > 0:
        bg_weight_matrix = weight_matrix * (preds < 0.5).float()
        bg_bound = torch.topk(bg_weight_matrix, bg_num, dim=-1)[0][:, :, -1:]
        bg_mask = (bg_weight_matrix >= bg_bound).float()
        bg_weight = bg_mask * weight_matrix
        # bg_max, _ = bg_weight.max(dim=-1)
        # bg_weight /= bg_max.unsqueeze(dim=-1)
        bg_weight_avg = bg_weight / bg_mask.sum(dim=-1).unsqueeze(dim=-1)
    else:
        bg_mask = weight_matrix.clone() * 0
        bg_weight = weight_matrix.clone() * 0.
        bg_weight_avg = bg_weight

    # top_num = math.ceil(H * W * top_thresh)
    # fg_sorted_ids = torch.argsort(weight_matrix * (preds > 0.5).float(), dim=-1)    # id 0 is the smallest similarity
    # fg_num = (preds[0, 0] > 0.5).float().sum()
    # fg_mask = fg_sorted_ids >= (H * W - fg_num)
    # bg_sorted_ids = torch.argsort(weight_matrix * (preds < 0.5).float(), dim=-1)  # id 0 is the smallest similarity
    # bg_num = (preds[0, 0] < 0.5).float().sum() * top_thresh
    # bg_mask = bg_sorted_ids >= (H * W - bg_num)
    # # bg_mask = sorted_ids < top_num

    fg_soft_label_init = ((fg_weight_avg * (preds - 0.5)).sum(dim=-1) + 0.5).reshape(B, H, W)
    bg_soft_label_init = ((bg_weight_avg * (preds - 0.5)).sum(dim=-1) + 0.5).reshape(B, H, W)

    # upsample soft labels
    up_sample = nn.Upsample(scale_factor=down_scale, mode='bilinear', align_corners=True)
    if label_channel == 'independent':  # not checked
        fg_soft_label = up_sample(fg_soft_label_init.unsqueeze(dim=1)).squeeze(dim=1)
        bg_soft_label = up_sample(bg_soft_label_init.unsqueeze(dim=1)).squeeze(dim=1)
        fg_soft_label = replace_labeled_pixels(fg_soft_label, targets, label='fg')
        bg_soft_label = replace_labeled_pixels(bg_soft_label, targets, label='bg')
        bg_soft_label = 1 - bg_soft_label
        soft_label = torch.cat((bg_soft_label.unsqueeze(dim=1), fg_soft_label.unsqueeze(dim=1)), dim=1)
        soft_label = nn.Softmax(dim=1)(soft_label)
        fg_mask_final = (soft_label[:, 0] < soft_label[:, 1]).float()
        bg_mask_final = 1 - fg_mask_final
        # softmax
    elif label_channel == 'dependent':
        fg_mask_final = ((fg_soft_label_init - 0.5) > (0.5 - bg_soft_label_init)).float()
        bg_mask_final = ((fg_soft_label_init - 0.5) <= (0.5 - bg_soft_label_init)).float()
        fg_soft_label = fg_soft_label_init * fg_mask_final + bg_soft_label_init * bg_mask_final
        fg_soft_label = up_sample(fg_soft_label.unsqueeze(dim=1)).squeeze(dim=1)
        fg_soft_label = replace_labeled_pixels(fg_soft_label, targets, label='fg_bg')
        bg_soft_label = 1 - fg_soft_label.clone()
        soft_label = torch.cat((bg_soft_label.unsqueeze(dim=1), fg_soft_label.unsqueeze(dim=1)), dim=1)
        # if i_batch == 5:
        #     ipdb.set_trace()
    else:
        raise NotImplementedError

    if opt.similarity.visualize:
        visualize_similarity_all(similarity_matrix.detach().clone().cpu().numpy(),
                                 distance_matrix.detach().clone().cpu().numpy(),
                                 weight_matrix.detach().clone().cpu().numpy(),
                                 fg_mask.detach().clone().cpu().numpy(),
                                 fg_weight.detach().clone().cpu().numpy(),
                                 fg_soft_label_init.detach().clone().cpu().numpy(),
                                 bg_mask.detach().clone().cpu().numpy(),
                                 bg_weight.detach().clone().cpu().numpy(),
                                 bg_soft_label_init.detach().clone().cpu().numpy(),
                                 soft_label.detach().clone().cpu().numpy(),
                                 fg_mask_final.detach().clone().cpu().numpy(),
                                 bg_mask_final.detach().clone().cpu().numpy(),
                                 targets.detach().clone().cpu().numpy(),
                                 preds_reserved.detach().clone().cpu().numpy(),
                                 inputs.detach().clone().cpu().numpy(), i_batch, B, H, W)

    return soft_label.detach()


def replace_labeled_pixels(probs, targets, label=''):
    if 'fg' in label:
        probs *= (1 - (targets == 1).float())
        probs += (targets == 1).float()
    if 'bg' in label:
        probs *= (1 - (targets == 0).float())
    return probs


def visualize_similarity_all(similarity_matrix, distance_matrix, weight_matrix, fg_mask, fg_weight, fg_soft_label,
                             bg_mask, bg_weight, bg_soft_label, soft_label, fg_mask_final, bg_mask_final,
                             targets, preds, inputs, i_batch, B, H, W):
    import matplotlib.pyplot as plt

    tmp_dir = args.weight_path.split('/')
    tmp_dir = tmp_dir[:-1] + ['visualize/batch_{}'.format(i_batch)]
    save_dir = '/' + tmp_dir[0]
    for t_dir in tmp_dir[1:]:
        save_dir = os.path.join(save_dir, t_dir)
    # save_dir = os.path.join(save_dir + 'visualize/batch_{}'.format(i_batch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if i_batch < 6:
    #     return
    # lis = [similarity_matrix, distance_matrix, weight_matrix, fg_mask, fg_weight, fg_soft_label,
    #                          bg_mask, bg_weight, bg_soft_label, soft_label, targets, preds, inputs]
    # for item in lis:
    #     print(item.shape)
    if i_batch > 20:
        ipdb.set_trace()
    for b in range(B):
        for i in range(H):
            for j in range(W):
                fig = plt.figure()
                ax = plt.subplot(3, 5, 1)
                ax.set_title('input image')
                tmp = copy.deepcopy(inputs[b, 0])
                plt.imshow(tmp)
                # plt.scatter(j, i, s=100, marker='.', c='r')

                ax = plt.subplot(3, 5, 2)
                ax.set_title('gt mask')
                tmp = copy.deepcopy(targets[b])
                tmp = tmp.astype(float)
                tmp[tmp == 255] = 0.5
                tmp[0, 0] = 0.99
                plt.imshow(tmp)

                ax = plt.subplot(3, 5, 3)
                ax.set_title('pred prob')
                tmp = copy.deepcopy(preds[b])
                tmp[0, 0] = 0.99
                plt.imshow(tmp)

                ax = plt.subplot(3, 5, 4)
                ax.set_title('similarity_matrix')
                pixel_id = i * W + j
                similarity = copy.deepcopy(similarity_matrix[b, pixel_id, :].reshape(H, W))
                similarity[0, 0] = 0.99
                plt.imshow(similarity)

                ax = plt.subplot(3, 5, 5)
                ax.set_title('distance')
                pixel_id = i * W + j
                distance = copy.deepcopy(distance_matrix[b, pixel_id, :].reshape(H, W))
                distance[0, 0] = 0.99
                plt.imshow(distance)

                ax = plt.subplot(3, 5, 6)
                ax.set_title('weight_matrix')
                pixel_id = i * W + j
                weight = copy.deepcopy(weight_matrix[b, pixel_id, :].reshape(H, W))
                weight[0, 0] = 0.99
                plt.imshow(weight)

                ax = plt.subplot(3, 5, 7)
                ax.set_title('fg_mask')
                pixel_id = i * W + j
                fg_mask_ij = copy.deepcopy(fg_mask[b, pixel_id, :].reshape(H, W))
                fg_mask_ij[0, 0] = 0.99
                plt.imshow(fg_mask_ij)

                ax = plt.subplot(3, 5, 8)
                ax.set_title('fg_weight')
                pixel_id = i * W + j
                fg_weight_ij = copy.deepcopy(fg_weight[b, pixel_id, :].reshape(H, W))
                fg_weight_ij[0, 0] = 0.99
                plt.imshow(fg_weight_ij)

                ax = plt.subplot(3, 5, 9)
                ax.set_title('bg_mask')
                pixel_id = i * W + j
                bg_mask_ij = copy.deepcopy(bg_mask[b, pixel_id, :].reshape(H, W))
                bg_mask_ij[0, 0] = 0.99
                plt.imshow(bg_mask_ij)

                ax = plt.subplot(3, 5, 10)
                ax.set_title('bg_weight')
                pixel_id = i * W + j
                bg_weight_ij = copy.deepcopy(bg_weight[b, pixel_id, :].reshape(H, W))
                bg_weight_ij[0, 0] = 0.99
                plt.imshow(bg_weight_ij)

                ax = plt.subplot(3, 5, 11)
                ax.set_title('fg_soft_label')
                pixel_id = i * W + j
                fg_soft_label_ij = copy.deepcopy(fg_soft_label[b])
                fg_soft_label_ij[0, 0] = 0.99
                plt.imshow(fg_soft_label_ij)

                ax = plt.subplot(3, 5, 12)
                ax.set_title('bg_soft_label')
                pixel_id = i * W + j
                bg_soft_label_ij = copy.deepcopy(bg_soft_label[b])
                bg_soft_label_ij[0, 0] = 0.99
                plt.imshow(bg_soft_label_ij)

                ax = plt.subplot(3, 5, 13)
                ax.set_title('soft_label')
                pixel_id = i * W + j
                soft_label_ij = copy.deepcopy(soft_label[b, 1, :])
                soft_label_ij[0, 0] = 0.99
                plt.imshow(soft_label_ij)

                ax = plt.subplot(3, 5, 14)
                ax.set_title('fg_mask_final')
                pixel_id = i * W + j
                fg_mask_final_ij = copy.deepcopy(fg_mask_final[b])
                fg_mask_final_ij[0, 0] = 0.99
                plt.imshow(fg_mask_final_ij)

                ax = plt.subplot(3, 5, 15)
                ax.set_title('bg_mask_final')
                pixel_id = i * W + j
                bg_mask_final_ij = copy.deepcopy(bg_mask_final[b])
                bg_mask_final_ij[0, 0] = 0.99
                plt.imshow(bg_mask_final_ij)

                plt.savefig(save_dir + '/{}_{}_{}.png'.format(b, i, j))
                plt.close('all')




if __name__ == '__main__':
    B, C, H, W = 2, 4, 4, 5
    features = torch.arange(B*C*H*W).reshape(B, C, H, W).float().cuda()
    preds = torch.arange(B*H*W).reshape(B, H, W).float().cuda() / 40
    targets = torch.zeros(B*H*W).reshape(B, H, W).cuda()
    targets[0, 1, 0] = 1
    targets[1, 1, 2] = 1
    targets[0, 0, :] = 255
    targets[1, :, 1] = 255
    soft_label = get_pseudo_mask(features, preds, targets)

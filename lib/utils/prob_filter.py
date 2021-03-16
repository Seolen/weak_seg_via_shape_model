import torch
import numpy as np
from lib.utils.gaussian import Gaussian2D
import torch.nn.functional as F
import ipdb
from datetime import datetime

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

def calculate_sdf(segs):
    '''
    segs: (B, D, H, W)
    :param segs:
    :return:
    '''
    sdfs = torch.zeros_like(segs, dtype=torch.float).cuda()
    batch = segs.shape[0]
    segs = segs.cpu().numpy()
    for ith in range(batch):
        seg = segs[ith]
        sdf = compute_sdf_np(seg)
        sdfs[ith] = torch.from_numpy(sdf).cuda()
    return sdfs

def calculate_pseudo_weight(prob, pseudo, method='sigmoid', gamma=2, mu=0.5):
    '''
    :param method: {'sigmoid', 'exponent'}
    :param mu:     only used of method 'sigmoid', the mean value, in [0, 1]
    :return:
    '''
    prob_flat = torch.flatten(prob)
    pseudo_flat = torch.flatten(pseudo)
    prob2 = torch.zeros_like(prob)

    pos_values, pos_indices = torch.sort(prob_flat[pseudo_flat == 1])
    neg_values, neg_indices = torch.sort(prob_flat[pseudo_flat == 0], descending=True)
    pos_rank_prob = torch.zeros_like(prob_flat[pseudo_flat == 1], dtype=torch.long)
    neg_rank_prob = torch.zeros_like(prob_flat[pseudo_flat == 0], dtype=torch.long)
    pos_rank_prob[pos_indices] = torch.arange(len(pos_indices), dtype=torch.long).cuda()
    neg_rank_prob[neg_indices] = torch.arange(len(neg_indices), dtype=torch.long).cuda()

    # Positive: ranking rescale [0,1];  Negative: uppper bound 2*Max_rank
    Max_rank = len(pos_rank_prob)
    pos_values = pos_rank_prob * 1.0 / Max_rank
    neg_rank_prob[neg_rank_prob > 2 * Max_rank] = 2 * Max_rank
    neg_values = neg_rank_prob * 1.0 / (2 * Max_rank)
    prob2[pseudo == 1] = pos_values
    prob2[pseudo == 0] = neg_values

    if method == 'sigmoid':
        tmp = (prob2 - mu) * gamma
        pseudo_weight = torch.sigmoid(tmp)
    else:
        pseudo_weight = torch.pow(prob2, gamma)
    return pseudo_weight

def calculate_rank(conf, mask, return_style='separate'):
    '''
    :param conf: confidence map of FG
    :param mask: mask map, 1 for FG region, 0 for BG region
    :param return_style: {'separate': separate FG, BG rank, 'union': the whole rank}
    :return:
    '''
    prob = conf; pseudo = mask
    prob_flat = torch.flatten(prob)
    pseudo_flat = torch.flatten(pseudo)
    prob2 = torch.zeros_like(prob)

    pos_values, pos_indices = torch.sort(prob_flat[pseudo_flat == 1])
    neg_values, neg_indices = torch.sort(prob_flat[pseudo_flat == 0], descending=True)
    pos_rank_prob = torch.zeros_like(prob_flat[pseudo_flat == 1], dtype=torch.long)
    neg_rank_prob = torch.zeros_like(prob_flat[pseudo_flat == 0], dtype=torch.long)
    pos_rank_prob[pos_indices] = torch.arange(len(pos_indices), dtype=torch.long).cuda()
    neg_rank_prob[neg_indices] = torch.arange(len(neg_indices), dtype=torch.long).cuda()

    # Positive: ranking rescale [0,1];  Negative: uppper bound 2*Max_rank
    # Max_rank = len(pos_rank_prob)
    # pos_values = pos_rank_prob * 1.0 / Max_rank
    # neg_rank_prob[neg_rank_prob > 2 * Max_rank] = 2 * Max_rank
    # neg_values = neg_rank_prob * 1.0 / (2 * Max_rank)

    pos_values, neg_values = pos_rank_prob, neg_rank_prob

    if return_style == 'separate':
        return pos_values, neg_values
    elif return_style == 'union':
        prob2[pseudo == 1] = pos_values
        prob2[pseudo == 0] = neg_values
        return prob2


def label_weigh_uncertain(probs, ae_preds=None, phase='ae_wce', weak_labels=None, method='sigmoid', gamma=2, mu=0.5):
    if weak_labels is not None:
        weak_labels = weak_labels[:, 0, ...]
    batch = probs.shape[0]
    pseudos = torch.zeros_like(probs, dtype=torch.long)
    pseudos_weights = torch.zeros_like(probs, dtype=torch.float)

    for ith in range(batch):
        prob, weak_label = probs[ith], weak_labels[ith]

        if phase == 'prob_wce':
            # 1. pseudo label
            pseudo = (prob > 0.5).long()
            if weak_labels is not None:
                pseudo[weak_label == 0] = 0
                pseudo[weak_label == 1] = 1

            # 2. pseudo weights: prob ranking
            pseudo_weight = calculate_pseudo_weight(prob, pseudo, method=method, gamma=gamma, mu=mu)

        elif phase == 'ae_wce':
            # 1. pseudo label: intersection as fg or bg, inconsistent region as unlabel
            ae_pred = ae_preds[ith]
            pred = (prob > 0.5).long()
            pseudo = torch.ones_like(ae_pred) * 255
            pseudo[(ae_pred == 1).long() + (pred == 1).long() == 2] = 1
            pseudo[(ae_pred == 0).long() + (pred == 0).long() == 2] = 0
            ## weak label reg
            if weak_labels is not None:
                pseudo[weak_label == 0] = 0
                pseudo[weak_label == 1] = 1

            # 2. pseudo weights: prob ranking
            pseudo_weight = calculate_pseudo_weight(prob, pseudo, method=method, gamma=gamma, mu=mu)

        pseudos[ith] = pseudo
        pseudos_weights[ith] = pseudo_weight

    return pseudos, pseudos_weights


class SlidingQueue():
    def __init__(self, size=500, mean=0.9731, std=0.0076):
        self.queue = torch.empty(size).normal_(mean=mean, std=std).cuda()

    def push_to_queue(self, x):
        self.queue = torch.cat((self.queue[1:], x))     #torch.tensor([x])
        return self.get_mean_std()

    def get_mean_std(self):
        return torch.mean(self.queue), torch.std(self.queue)


def label_filter_uncertain(probs, phase='prob', sdfs=None, ae_labels=None, ae_probs=None, weak_labels=None,
                           fp_filter_ratio=0.5, fn_filter_ratio=2.0, filter_k_times=0, psrk=1.0,
                           weak_logits=None, ae_logits=None, t_scaling_temperatures=[1, 1], t_scaling_combine='add',
                           extra_mode=''):
    '''
    :param probs: shape (B, D, H, W)
    :param phase: {prob, ae, sdf, sdfbg, probae, probsdf}
    :param fp_filter_ratio:
    :param fn_filter_ratio:
    :param psrk: combine weight of prob and sdf ranking (rank_prob + k*rank_sdf)
    :return: pseudos with deep supervision (a list)
    '''

    pseudos = torch.zeros_like(probs, dtype=torch.long).cuda()
    batch = probs.shape[0]
    # params for 'probsdf'
    if phase == 'probsdf':
        mean, variance = (0.5, 0), (1, 1)   #(0.5 / 3, 1.0 / 3)
        gauss = Gaussian2D(mean, variance)
    elif phase == 't_scaling':
        temperatures, combine_mode = t_scaling_temperatures, t_scaling_combine
        weak_logits, ae_logits = weak_logits / temperatures[0], ae_logits / temperatures[1]
        probs, ae_probs = F.softmax(weak_logits, dim=1)[:, 1, ...], F.softmax(ae_logits, dim=1)[:, 1, ...]

    for ith in range(batch):
        prob, weak_label = probs[ith], weak_labels[ith, 0]
        pred = (prob > 0.5).long()
        pseudo = pred.clone()

        if phase == 'prob' or phase == 'ae':
            # filter to pred
            prob_flat = torch.flatten(prob)
            pos_values, _ = torch.sort(prob_flat[prob_flat > 0.5])
            neg_values, _ = torch.sort(prob_flat[prob_flat <= 0.5], descending=True)
            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if filter_k_times > 0:
                    #TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob < pos_value).long() == 2] = 255  # pseudo[torch.logical_and(pseudo == 1, prob < pos_value)] = 255
                pseudo[(pseudo == 0).long() + (prob > neg_value).long() == 2] = 255 # pseudo[torch.logical_and(pseudo == 0, prob > neg_value)] = 255
            else:
                pseudo = weak_label.clone()

        elif phase == 't_scaling':
            # temperature scaling have been done
            # mask out region
            ae_prob = ae_probs[ith]; ae_label = ae_labels[ith]
            pseudo[(ae_label == 1).long() + (pred == 0).long() == 2] = 255
            pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 255

            # prob filter
            if t_scaling_combine == 'add':
                prob_ = torch.add(prob, ae_prob)
            elif t_scaling_combine == 'mul':
                prob_ = torch.mul(prob, ae_prob)
            elif t_scaling_combine == 'add0.5':
                prob_ = torch.add(prob, 0.5*ae_prob)
            elif t_scaling_combine == 'add2':
                prob_ = torch.add(prob, 2*ae_prob)

            prob_flat = torch.flatten(prob_)
            pseudo_flat = torch.flatten(pseudo)
            pos_values, _ = torch.sort(prob_flat[pseudo_flat == 1])
            neg_values, _ = torch.sort(prob_flat[pseudo_flat == 0], descending=True)
            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob_ < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob_ > neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()

            # extra mode
            if 'trust_aebg' in extra_mode:
                pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 0


        elif phase == 'ae_psrank':
            # fg, bg filter with combined ranking of ae prob and tsdf
            prob_flat = torch.flatten(prob)
            sdf = sdfs[ith]; sdf_flat = torch.flatten(sdf)
            prob2 = torch.zeros_like(prob)

            pos_values, pos_indices = torch.sort(prob_flat[prob_flat > 0.5])
            neg_values, neg_indices = torch.sort(prob_flat[prob_flat <= 0.5], descending=True)
            pos_rank_prob = torch.zeros_like(prob_flat[prob_flat > 0.5], dtype=torch.long); pos_rank_sdf = pos_rank_prob.clone()
            neg_rank_prob = torch.zeros_like(prob_flat[prob_flat <= 0.5], dtype=torch.long); neg_rank_sdf = neg_rank_prob.clone()
            pos_rank_prob[pos_indices] = torch.arange(len(pos_indices), dtype=torch.long).cuda()
            neg_rank_prob[neg_indices] = torch.arange(len(neg_indices), dtype=torch.long).cuda()
            pos_values, pos_indices = torch.sort(sdf_flat[prob_flat > 0.5])
            neg_values, neg_indices = torch.sort(sdf_flat[prob_flat <= 0.5], descending=True)
            pos_rank_sdf[pos_indices] = torch.arange(len(pos_indices), dtype=torch.long).cuda()
            neg_rank_sdf[neg_indices] = torch.arange(len(neg_indices), dtype=torch.long).cuda()
            pos_score = pos_rank_prob + psrk * pos_rank_sdf
            neg_score = neg_rank_prob + psrk * neg_rank_sdf
            pos_values, _ = torch.sort(pos_score)
            neg_values, _ = torch.sort(neg_score)
            prob2[prob > 0.5] = pos_score; prob2[prob <= 0.5] = neg_score

            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if filter_k_times > 0:
                    # TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob2 < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob2 < neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()

        elif phase == 'sdf' or phase == 'sdfbg':
            # fg, bg filter with sdf
            sdf = sdfs[ith]
            sdf_flat = torch.flatten(sdf)
            if phase == 'sdf':
                pos_values, _ = torch.sort(sdf_flat[prob_flat > 0.5])
            elif phase == 'sdfbg':
                pos_values, _ = torch.sort(prob_flat[prob_flat > 0.5])
            neg_values, _ = torch.sort(sdf_flat[prob_flat <= 0.5], descending=True)  # sdf is negative for BG

            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if filter_k_times > 0:
                    # TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]

                if phase == 'sdf':
                    pseudo[(pseudo == 1).long() + (sdf_flat < pos_value).long() == 2] = 255
                elif phase == 'sdfbg':
                    pseudo[(pseudo == 1).long() + (prob < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob > neg_value).long() == 2] = 255
                pseudo = weak_label.clone()

        elif phase == 'probae':
            # mask out region
            ae_label = ae_labels[ith]
            pseudo[(ae_label == 1).long() + (pred == 0).long() == 2] = 255
            pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 255
            # prob filter
            prob_flat = torch.flatten(prob)
            pseudo_flat = torch.flatten(pseudo)
            pos_values, _ = torch.sort(prob_flat[pseudo_flat == 1])
            neg_values, _ = torch.sort(prob_flat[pseudo_flat == 0], descending=True)
            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if filter_k_times > 0:
                    # TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob > neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()

            # extra mode
            if 'trust_aebg' in extra_mode:
                pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 0


        elif phase == 'probae_adaptive':
            # mask out region
            ae_label = ae_labels[ith]
            pseudo[(ae_label == 1).long() + (pred == 0).long() == 2] = 255
            pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 255
            # prob filter
            prob_flat = torch.flatten(prob)
            pseudo_flat = torch.flatten(pseudo)
            pos_values, _ = torch.sort(prob_flat[pseudo_flat == 1])
            neg_values, _ = torch.sort(prob_flat[pseudo_flat == 0], descending=True)

            '''
            # TODO: adaptive fpr
            def calculate_fpr(avg_prob, sq_mean, sq_std,
                              mean_fpr=0.5, min_fpr=0.3, max_fpr=0.7):
                slope = (mean_fpr - max_fpr) / (3 * sq_std)
                current_fpr = mean_fpr + slope * (avg_prob - sq_mean)
                if current_fpr < min_fpr:
                    return min_fpr
                elif current_fpr > max_fpr:
                    return max_fpr
                return current_fpr

            avg_prob = (prob[prob > 0.5]).sum() / (prob > 0.5).sum()
            avg_prob = torch.tensor([avg_prob]).cuda()
            sq_mean, sq_std = sliding_queue.push_to_queue(avg_prob)
            fp_filter_ratio = calculate_fpr(avg_prob, sq_mean, sq_std)
            '''

            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob > neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()

            # extra mode
            if 'trust_aebg' in extra_mode:
                pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 0




        elif phase == 'probae_addrank':
            # total rank = prob rank + ae rank, then filter according to the ratio
            # mask out region
            prob2 = torch.zeros_like(prob, dtype=torch.long)
            ae_label = ae_labels[ith]; ae_prob = ae_probs[ith]
            pseudo[(ae_label == 1).long() + (pred == 0).long() == 2] = 255
            pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 255
            # prob rank + ae rank
            prob_rank_pos, prob_rank_neg = calculate_rank(prob, pseudo)
            aepr_rank_pos, aepr_rank_neg = calculate_rank(ae_prob, pseudo)
            pos_score = prob_rank_pos + aepr_rank_pos
            neg_score = prob_rank_neg + aepr_rank_neg
            pos_values, _ = torch.sort(pos_score)
            neg_values, _ = torch.sort(neg_score)
            prob2[pseudo == 1] = pos_score; prob2[pseudo == 0] = neg_score

            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (prob2 < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (prob2 < neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()
            if 'trust_aebg' in extra_mode:
                pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 0


        elif phase == 'aeprob':
            # mask out region
            ae_label = ae_labels[ith]; ae_prob = ae_probs[ith]
            pseudo[(ae_label == 1).long() + (pred == 0).long() == 2] = 255
            pseudo[(ae_label == 0).long() + (pred == 1).long() == 2] = 255
            # prob filter
            prob_flat = torch.flatten(ae_prob)
            pseudo_flat = torch.flatten(pseudo)
            pos_values, _ = torch.sort(prob_flat[pseudo_flat == 1])
            neg_values, _ = torch.sort(prob_flat[pseudo_flat == 0], descending=True)
            if len(pos_values) != 0 and len(neg_values) != 0:  # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                if filter_k_times > 0:
                    # TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (ae_prob < pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (ae_prob > neg_value).long() == 2] = 255
            else:
                pseudo = weak_label.clone()

        elif phase == 'probsdf':
            # calculate uncertainty
            sdf = sdfs[ith]
            uncertainty = gauss.forward(prob, sdf)

            # probsdf filter
            prob_flat = torch.flatten(prob)
            uncertainty_flat = torch.flatten(uncertainty)
            pos_values, _ = torch.sort(uncertainty_flat[prob_flat > 0.5], descending=True)
            neg_values, _ = torch.sort(uncertainty_flat[prob_flat <= 0.5], descending=True)
            if len(pos_values) != 0 and len(neg_values) != 0:    # FG region exists
                out_num = int(pos_values.shape[0] * fp_filter_ratio)
                pos_value = pos_values[out_num]
                out_num2 = int(out_num * fn_filter_ratio)
                if filter_k_times > 0:
                    # TODO: discard filter_k_times, it is unreasonable.
                    neg_num = filter_k_times * pseudo.sum() - out_num
                    neg_value = neg_values[int(neg_num)]
                elif int(out_num * fn_filter_ratio) <= len(neg_values):
                    neg_value = neg_values[int(out_num * fn_filter_ratio)]
                else:
                    neg_value = neg_values[int(len(neg_values) * 0.5)]
                pseudo[(pseudo == 1).long() + (uncertainty > pos_value).long() == 2] = 255
                pseudo[(pseudo == 0).long() + (uncertainty > neg_value).long() == 2] = 255
                # sdf refine: sdf to filter FP: (pred==1, sdf<0)
                pseudo[(pseudo == 1).long() + (sdf < 0).long() == 2] = 0
            else:
                pseudo = weak_label.clone()

        # weak reg pseudo
        if extra_mode != 'disable_weak_add_fgbg':
            pseudo[weak_label == 0] = 0
            if extra_mode not in ['disable_weak_add_fg', 'modify_weak_label', 'modify_initial_weak']:
                pseudo[weak_label == 1] = 1
        pseudos[ith] = pseudo

    return pseudos


def downsample_seg(segmentation, out_shape):
    '''
    :param segmentation: (B, D, H, W)
    :param out_shape: (d, h, w)
    :return:
    '''
    # downsampling
    seg = segmentation.unsqueeze(dim=1).float()
    seg = torch.nn.functional.interpolate(seg, size=out_shape, mode='nearest')
    return seg.long()

def downsample_seg_scales(segmentation, out_shapes):
    '''
    :param segmentation: (B, D, H, W)
    :param out_shapes: [(d1, h1, w1), (d2, h2, w2), ...]
    :return:
    '''
    outputs = []
    for out_shape in out_shapes:
        out_shape = out_shape[-3: ]
        outputs.append(downsample_seg(segmentation, out_shape))
    return outputs



# main
# sliding_queue = SlidingQueue()
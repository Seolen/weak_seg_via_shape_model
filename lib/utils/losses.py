import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.configs.parse_arg import opt, args
import ipdb

'''
input shape:
    pred, gt: (B, C, H, W, ...), (B, H, W, ...)
'''


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none')
        # self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target, multi_loss=False):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target_new = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target_new.unsqueeze(1)).squeeze(dim=1)
        # ipdb.set_trace()#a 0.25, loss*=a
        # TODO: how about 255
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        loss = loss[target != self.ignore_index].mean()

        if multi_loss:
            with torch.no_grad():
                bg_loss = cross_entropy[target==0].mean()
                fg_loss = cross_entropy[target==1].mean()
            # # bg loss
            # gt = target.clone()
            # gt[target > 0] = self.ignore_index
            # num = (gt != self.ignore_index).sum()
            # with torch.no_grad():
            #     bg_loss = super().forward(input_, gt)
            #     bg_loss = bg_loss.sum() / num

            # # fg loss
            # gt = target.clone()
            # gt[target == 0] = self.ignore_index
            # num = (gt != self.ignore_index).sum()
            # with torch.no_grad():
            #     fg_loss = super().forward(input_, gt)
            #     fg_loss = fg_loss.sum() / num

            return [loss, fg_loss, bg_loss], ['pce_loss', 'fg_loss', 'bg_loss']
        else:
            return loss


class RegLoss(nn.Module):
    def __init__(self, loss_func, ignore_index=255):
        super(RegLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_func = eval(loss_func)(reduction='none')
        # self.loss_func = nn.MSELoss(reduction='none')
        # self.loss_func = nn.L1Loss(reduction='none')
    
    def forward(self, inputs, targets):
        loss = self.loss_func(inputs[:, 0], targets.float())
        mask = (targets != self.ignore_index)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class CELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=torch.Tensor([1.0, 1.0]).cuda(), ignore_index=255, multi_loss=True, pixel_weights=None):
        self.multi_loss = multi_loss
        self.ignore_index = ignore_index
        self.pixel_weights = pixel_weights
        self.given_weight = weight
        if pixel_weights is not None:
            super().__init__(ignore_index=ignore_index, reduction='none')
        else:
            super().__init__(weight=weight, ignore_index=ignore_index)

    def forward(self, input, target, multi_loss=False):
        ce_loss = super().forward(input, target)
        if self.pixel_weights is not None:
            ce_loss = torch.mul(ce_loss, self.pixel_weights)
            # ce_loss = ce_loss[target != self.ignore_index].mean()
            # hand-written weighted
            fg_sum, bg_sum = (target==1).long().sum(), (target==0).long().sum()
            fg_weight, bg_weight = self.given_weight[1], self.given_weight[0]
            ce_loss = (fg_weight * ce_loss[target==1].sum() + bg_weight * ce_loss[target==0].sum()) / (fg_sum*fg_weight + bg_sum*bg_weight)

        if multi_loss:  #self.multi_loss:
            # bg loss
            gt = target.clone()
            gt[target > 0] = self.ignore_index
            with torch.no_grad():
                bg_loss = super().forward(input, gt)

            # fg loss
            gt = target.clone()
            gt[target == 0] = self.ignore_index
            with torch.no_grad():
                fg_loss = super().forward(input, gt)

            return [ce_loss, fg_loss, bg_loss], ['pce_loss', 'fg_loss', 'bg_loss']

        else:
            return ce_loss


class SoftCELoss(object):
    def __init__(self, auto_weight=False, weight=torch.Tensor([1.0, 1.0]).cuda()):
        self.auto_weight = auto_weight
        self.weight = weight

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputs, targets):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        B, C, H, W = targets.shape
        sample_num = B * H * W
        bg_mask = (targets[:, 0:1, :, :] > 0.5).float()
        fg_mask = (targets[:, 1:2, :, :] > 0.5).float()
        if self.auto_weight:
            bg_loss = torch.sum(torch.mul(log_likelihood, targets * bg_mask)) / bg_mask.sum()
            fg_loss = torch.sum(torch.mul(log_likelihood, targets * fg_mask)) / fg_mask.sum()
            pce_loss = bg_loss + fg_loss
        else:
            bg_loss = torch.sum(torch.mul(log_likelihood, targets * bg_mask)) / sample_num
            fg_loss = torch.sum(torch.mul(log_likelihood, targets * fg_mask)) / sample_num
            pce_loss = bg_loss * self.weight[0] + fg_loss * self.weight[1]

        return [pce_loss, fg_loss, bg_loss], ['pce_loss', 'fg_loss', 'bg_loss']


class DiceLoss(object):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, score, target, smooth=1e-8):
        target = target.float()

        # filter ignored region
        if self.ignore_index is not None:
            score = score[target != self.ignore_index]
            target = target[target != self.ignore_index]

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return [loss], ['dice_loss']


class CEDiceLoss(object):
    def __init__(self, weight=torch.Tensor([1.0, 1.0]).cuda(), ignore_index=255, multi_loss=True):
        self.multi_loss = multi_loss
        self.ignore_index = ignore_index
        self.celoss = CELoss(weight, ignore_index)
        self.dice_loss = DiceLoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target, multi_loss=False):
        pred = nn.Softmax(dim=1)(input)[:, 1, ...]
        if multi_loss:  #self.multi_loss:
            ce_losses, ce_losses_names = self.celoss(input, target, multi_loss)
            dice_loss, dice_name = self.dice_loss(pred, target)
            total_loss, total_name = ce_losses[0] + dice_loss[0], 'total_loss'
            return [total_loss]+ce_losses+dice_loss, [total_name]+ce_losses_names+dice_name
        else:
            ce_loss = self.celoss(input, target, multi_loss)
            dice_loss, dice_name = self.dice_loss(pred, target)
            total_loss = ce_loss + dice_loss[0]
            return total_loss



########################## Metrics #####################

class DiceMetric(object):
    def __init__(self, dice_each_class=False, smooth=1e-8):
        self.dice_each_class = dice_each_class
        self.smooth = smooth

    def forward(self, preds, gts):
        # preds, gts = preds.detach(), gts.detach()

        # fg dice
        dice = 0
        batch = preds.shape[0]
        for ith in range(batch):
            dice += self.dice_func(preds[ith], gts[ith])
        dice /= batch
        dice_fg = dice

        # bg dice
        if self.dice_each_class:
            dice = 0
            batch = preds.shape[0]
            for ith in range(batch):
                dice += self.dice_func(preds[ith], gts[ith], type='bg')
            dice /= batch
            dice_bg = dice

            return [dice_fg, dice_bg], ['dice', 'dice_bg']
        return [dice_fg], ['dice']


    def dice_func(self, pred, gt, type='fg'):
        if type == 'fg':
            pred = pred > 0.5
            label = gt > 0
        else:
            pred = pred < 0.5
            label = gt == 0
        inter_size = torch.sum(((pred * label) > 0).float())
        sum_size = (torch.sum(pred) + torch.sum(label)).float()
        dice = (2 * inter_size + self.smooth) / (sum_size + self.smooth)
        return dice


########################## nnUNet Losses #####################

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            # self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)
            raise NotImplementedError
    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0

        # print('ce_loss', ce_loss)
        # print('dice_loss', dc_loss)

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss=None, weight_factors=None, ce_loss=False, multi_loss=False, regress=False):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.ce_loss = ce_loss
        self.multi_loss = multi_loss
        self.regress = regress
    
    def compute_loss(self, x, y, multi_loss=False, pixel_weights=None, disable_auto_weight=False):
        eps = 1e-6
        if (not disable_auto_weight) and opt.train.auto_weight_fg_bg and (not self.regress):
            loss = []
            for b in range(x.shape[0]):
                fg_weight = ((y[b:b+1] == 0).sum().float() / ((y[b:b+1] == 1).sum().float() + eps)).item()
                if fg_weight > opt.train.max_fg_weight:
                    fg_weight = opt.train.max_fg_weight
                class_weights = torch.Tensor([1.0, fg_weight]).cuda() # only for weak now
                if pixel_weights is not None:
                    loss_func = CELoss(weight=class_weights, ignore_index=255, pixel_weights=pixel_weights[b:b + 1])
                else:
                    loss_func = CELoss(weight=class_weights, ignore_index=255, pixel_weights=pixel_weights)
                loss.append(loss_func(x[b:b+1], y[b:b+1]))
            return sum(loss) / len(loss)
        else:
            if multi_loss:
                return self.loss(x, y, multi_loss)
            else:
                return self.loss(x, y)

    def forward(self, x, y, pixel_weights=None, disable_auto_weight=False):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        # shuailin add
        losses = []; loss_names = []
        if self.ce_loss:
            y = [it[:, 0, ...] for it in y]     # no channel dim in CELoss
        
        if self.multi_loss:
            l = self.compute_loss(x[0], y[0], self.multi_loss)
            assert isinstance(l, (tuple))
            assert isinstance(l[0], (list)) and isinstance(l[1], (list))
            losses, loss_names = [weights[0] * item for item in l[0]], ['d0'] + l[1][1:]
            l = losses[0]
        else:
            l = weights[0] * self.compute_loss(x[0], y[0], pixel_weights=pixel_weights, disable_auto_weight=disable_auto_weight)
            losses.append(l); loss_names.append('d0')
        for i in range(1, len(x)):
            if weights[i] != 0:
                loss = weights[i] * self.compute_loss(x[i], y[i], pixel_weights=pixel_weights, disable_auto_weight=disable_auto_weight)
                l = l + loss
                losses.append(loss); loss_names.append('d%d' % i)

        # return l
        return [l]+losses, ['loss']+loss_names


if __name__ == '__main__':

    '''
    # softmax_helper is the same as nn.Softmax(), can we replace it?
    a = torch.tensor([[1.0, 2.0]])
    print(softmax_helper(a))
    print(nn.Softmax(dim=1)(a))
    '''

    '''
    Input: params{pool_op_kernel_sizes...}, inputs, targets
    Output: loss scalar
    Status:
        - verified no bug in the loss pipeline
        - Not verify the correctness of each loss term {dice_loss, ce_loss}
        - Not return each separate loss, for convenient loss curve  
    
    Example:
    
    '''

    # loss weights
    pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    ## below from nnUNet code
    net_numpool = len(pool_op_kernel_sizes)
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
    mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    print('weights:\t', weights)

    # loss params
    batch_dice = True
    loss_func = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
    loss = MultipleOutputLoss2(loss_func, weights)

    # input and target
    input = [torch.tensor(
        [[0.6, 0.3],
         [1.2, 0.1]]
        ).float()]
    target = [torch.tensor(
        [1, 0]
        ).long()]
    loss_value = loss(input, target)
    print(loss_value)












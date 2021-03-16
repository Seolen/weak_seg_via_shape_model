#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ipdb
# import sys
# import os
# sys.path.append(os.getcwd())    # comment when no longer debug

from copy import deepcopy
from torch import nn
import torch
import numpy as np
from lib.models.initialization import InitWeights_He   #
from lib.models.neural_network import SegmentationNetwork  #
# from initialization import InitWeights_He   #
# from neural_network import SegmentationNetwork  #
import torch.nn.functional

# from non_local.non_local_dot_product import NONLocalBlock3D
from lib.models.non_local.non_local_dot_product import NONLocalBlock3D


def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,
                 gnn_type=None, n_blocks=0, resolution=0, sub_sample=False, bn_layer=False,
                 return_feature=False, disable_skip=False,
                 sp_use_type=''):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.return_feature = return_feature
        self.disable_skip = disable_skip

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]

            # shuailin: skip choice
            if not disable_skip:
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                      self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                      self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                    StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                      self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                      self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
                ))
            else:
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                      self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                      self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
                ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        self.resolution = resolution
        self.gnn_type = gnn_type
        if self.gnn_type == 'non_local':
            non_local_in_channels = {24: 256, 12: 320, 6: 320}  # can cause error if resolution not in this dict.
            non_local_in_channel = non_local_in_channels[abs(resolution)]
            self.non_local_block = NONLocalBlock3D(non_local_in_channel, sub_sample=sub_sample, bn_layer=bn_layer)
            
            # self.non_local_blocks = nn.ModuleList(non_local_blocks)

        self.sp_use_type = sp_use_type
        if self.sp_use_type != '':
            self.sp_tu = deepcopy(self.tu)
            self.sp_conv_blocks_localization = deepcopy(self.conv_blocks_localization)
            if 'regress' in self.sp_use_type:
                self.sp_seg_outputs = []
                for ds in range(len(self.conv_blocks_localization)):
                    self.sp_seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, 1,
                                                    1, 1, 0, 1, 1, seg_output_use_bias))
                self.sp_seg_outputs = nn.ModuleList(self.sp_seg_outputs)
            else:
                self.sp_seg_outputs = deepcopy(self.seg_outputs)
            self.sp_final_nonlin = deepcopy(self.final_nonlin)
            if 'fuse' in self.sp_use_type:
                self.sp_fuse_conv_blocks_localization = deepcopy(self.conv_blocks_localization)

    def forward(self, x, feat_for_pseudo=None):
        if self.sp_use_type != '':
            out_all = self.forward_superpixel(x)
        else:
            out_all = self.forward_before_0818(x, feat_for_pseudo)
        return out_all

    def forward_superpixel(self, x):
        skips = []
        seg_outputs = []
        sp_outputs = []
        sp_features = []
        # down size
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:  #False
                x = self.td[d](x)
        bottle_neck = self.conv_blocks_context[-1](x)

        # up size 2, sp
        x = bottle_neck
        if self.sp_use_type != '':
            for u in range(len(self.sp_tu)):
                x = self.sp_tu[u](x)
                sp_features.append(x)
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
                x = self.sp_conv_blocks_localization[u](x)
                sp_outputs.append(self.sp_final_nonlin(self.sp_seg_outputs[u](x)))  #sp_final_nonlin: indentity mapping

        # up size 1, seg
        x = bottle_neck
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            if 'fuse_before' in self.sp_use_type:
                x = torch.cat((x, sp_features[u]), dim=1)
                x = self.sp_fuse_conv_blocks_localization[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            if u == len(self.tu)-1 and self.return_feature:
                last_feature = x
            x = self.conv_blocks_localization[u](x)
            if 'fuse_after' in self.sp_use_type:
                x = torch.cat((x, sp_features[u]), dim=1)
                x = self.sp_fuse_conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))   #final_nonlin: indentity mapping
        
        if self._deep_supervision and self.do_ds:
            out_all = tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            out_all = seg_outputs[-1]

        if self.return_feature:
            out_all = (last_feature, out_all)
        if self.sp_use_type != '':
            if self._deep_supervision and self.do_ds:
                assert not self.upscale_logits
                out_all = (out_all, tuple([sp_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], sp_outputs[:-1][::-1])]))
            else:
                out_all = (out_all, sp_outputs[-1])
        return out_all

        # if self._deep_supervision and self.do_ds:
        #     if self.return_feature:
        #         return last_feature, tuple([seg_outputs[-1]] + [i(j) for i, j in
        #                                       zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        #     return tuple([seg_outputs[-1]] + [i(j) for i, j in
        #                                       zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        # else:
        #     if self.return_feature:
        #         return last_feature, seg_outputs[-1]
        #     return seg_outputs[-1]
    
    def forward_before_0818(self, x, feat_for_pseudo=None):
        skips = []
        seg_outputs = []
        if feat_for_pseudo is not None:
            last_feature = []
            if 'intensity' in feat_for_pseudo:
                last_feature.append(x)
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            if self.resolution == -x.shape[-1]:
                if self.gnn_type == 'non_local':
                    x = self.non_local_block(x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        if feat_for_pseudo is not None:
            if 'first' in feat_for_pseudo:
                last_feature.append(skips[0])
        x = self.conv_blocks_context[-1](x)
        if self.resolution == x.shape[-1]:
            if self.gnn_type == 'non_local':
                x = self.non_local_block(x)
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            if not self.disable_skip:
                x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if u == len(self.tu)-1 and (self.return_feature or feat_for_pseudo is not None):
                if feat_for_pseudo is not None:
                    if 'last' in feat_for_pseudo:
                        last_feature.append(x)
                if self.return_feature:
                    last_feature = x
                    ipdb.set_trace() # the feature dim was 64 before, possible error
            if self.resolution == x.shape[-1]:
                if self.gnn_type == 'non_local':
                    x = self.non_local_block(x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
        if self._deep_supervision and self.do_ds:
            if self.return_feature or feat_for_pseudo is not None:
                return last_feature, tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            if self.return_feature or feat_for_pseudo is not None:
                return last_feature, seg_outputs[-1]
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp



if __name__ == '__main__':
    '''
    Generic_UNet:
        Params: pool_op_kernel_sizes::list, conv_kernel_sizes::list
        Note: 
        - deep supervision,     which means the network output [1+len(pool_ops)-1] logits maps.
        - input size,    must be N times of pool_op_kernel_sizes, no round operation in network.
        - output logits,    current no softmax or sigmoid employed, thus data value in [-K, +J]
    
    Below example are from F.2 and Figure F.1(page35), PROMISE12(page44).
    Example 1: Promise (Prostate)
        - input shape: (2, 1, 20, 256, 224)
        - output:
            Deep supervision is used. The output is a list with 5 logits maps
            Shapes of output logits are as follows:
                torch.Size([2, 2, 20, 256, 224])
                torch.Size([2, 2, 20, 128, 112])
                torch.Size([2, 2, 10, 64, 56])
                torch.Size([2, 2, 5, 32, 28])
                torch.Size([2, 2, 5, 16, 14])
    Example 2: Segthor-Heart
        - input shape: (2, 1, 64, 192, 160)
        - output:
            Deep supervision is used. The output is a list with 5 logits maps
            Shapes of output logits are as follows:
                torch.Size([2, 2, 64, 192, 160])
                torch.Size([2, 2, 64, 96, 80])
                torch.Size([2, 2, 32, 48, 40])
                torch.Size([2, 2, 16, 24, 20])
                torch.Size([2, 2, 8, 12, 10])
    '''


    test_type = 'Promise'   #'Segthor' #

    if test_type == 'Promise':
        # PROMISE12 setting
        pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]          # downsample [4, 32, 32]
        conv_kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        input_shape = (2, 1, 20, 256, 224)
        pool_op_kernel_sizes = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        conv_kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        # input_shape = (2, 1, 40, 192, 192)
        input_shape = (1, 1, 40, 160, 160)
    else:
        # Segthor-Heart setting
        pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]      # downsample [16, 32, 32]
        conv_kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        input_shape = (2, 1, 64, 192, 160)


    net_params={
        'input_channels': 1, 'base_num_features': 32, 'num_classes': 2,
        'num_pool': len(pool_op_kernel_sizes),

        'num_conv_per_stage': 2, 'feat_map_mul_on_downscale': 2, 'conv_op': nn.Conv3d,
        'norm_op': nn.InstanceNorm3d, 'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': nn.Dropout3d, 'dropout_op_kwargs': {'p': 0, 'inplace': True},
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        'deep_supervision': True, 'dropout_in_localization': False, 'final_nonlin': lambda x: x,
        'weightInitializer': InitWeights_He(1e-2),

        'pool_op_kernel_sizes': pool_op_kernel_sizes,
        'conv_kernel_sizes': conv_kernel_sizes,
        'upscale_logits': False, 'convolutional_pooling': True, 'convolutional_upsampling': True,
        'gnn_type': 'non_local', 'n_blocks': 1, 'resolution': 24,
        'sub_sample': False, 'bn_layer': False,
        'sp_use_type': '', #''decoder_fuse_before'

        'disable_skip': True,
    }

    net = Generic_UNet(**net_params)
    input = torch.randn(input_shape).float().cuda()
    net = net.cuda()
    out = net(input)
    ipdb.set_trace()

    print('Deep supervision is used. The output is a list with %d logits maps' % len(out))
    print('Shapes of output logits are as follows:')
    for term in out:
        print(term.shape)

    ipdb.set_trace()
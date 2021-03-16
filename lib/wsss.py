import os
import sys
import copy
import time
import datetime
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lib.models import HNN1, HNN2, VNet, ResidualUNet, Generic_UNet, AutoEnc
from lib.datasets import BGDataset, get_moreDA_augmentation, AEDataset, get_moreDA_augmentation_ae
from lib.configs.parse_arg import opt, args
from lib.utils import LossMeter, MultiLossMeter, CELoss, DiceMetric, CEDiceLoss, MultipleOutputLoss2,\
    DC_and_CE_loss, FocalLoss, RegLoss, label_filter_uncertain, downsample_seg_scales, calculate_sdf,\
    label_weigh_uncertain, SlidingQueue
from lib.vis.util import save_pred
import ast
from scipy import ndimage
from scipy.io import savemat
import nibabel as nib

# import imcut.pycut as pspc
import ipdb

# import random
# import torch.backends.cudnn as cudnn
#
# if args.deterministic:
#     cudnn.benchmark = False
#     cudnn.deterministic = True
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)


class WSSS(object):
    def __init__(self):
        super(WSSS, self).__init__()
        self.since = time.time()
        self.phases = []
        self.val_ids = None
        self.ae_model = None
        self.pseudo_criterion = None    # independent pseudo criterion or default the same

        self.EM_save = False
        if opt.model.iterate_epoch > 0 or args.save_em_init \
                or opt.model.filter_extra == 'modify_initial_weak':     # need save pseudo label in E step
            self.E_train_loader = None
            self.EM_save = True

        self.model = self.build_model()
        self.criterion, self.metric, self.optimizer, self.scheduler = self.build_optimizer()
        self.data_loaders, self.data_sizes, self.datasets = self.build_dataloader()

        self.best = {'loss': 1e8, 'dice': 0, 'epoch': -1, 'dice_final': 0}
        self.log_init()
        if (not args.save_em_init) and (args.demo == ''):
            self.writer = SummaryWriter(opt.tb_dir)

        self.save_pred = save_pred

    def build_dataloader(self):
        if args.demo == '':
            self.phases = ['train', 'val']
            candidates = ['Prostate_2D', 'Prostate_2DE', 'Heart']
            if opt.data.dataset in candidates:
                dataset = eval(opt.data.dataset)
                data_loaders = {
                    'train': DataLoader(dataset(data_dir=opt.data.data_dir, weak=opt.data.use_weak, phase='train'),
                        batch_size=opt.train.train_batch, shuffle=True, num_workers=4, pin_memory=True),
                    'val': DataLoader(dataset(data_dir=opt.data.data_dir, weak=opt.data.use_weak, phase='val'),
                                      batch_size=opt.train.valid_batch, num_workers=4, pin_memory=True),
                }
            elif opt.data.dataset == 'BGDataset':
                # data dir
                data_dir = opt.data.data_dir

                # use select samples
                if not opt.data.use_select:
                    use_select = False
                    select_params = None
                else:
                    use_select = True
                    select_params = {'label_percent': opt.data.select_label_percent,
                                     'use_pred_num': opt.data.select_use_pred_num}

                # dataset
                ds_train = BGDataset(data_dir, use_weak=opt.data.use_weak, phase='train',
                                     batch_size=opt.train.train_batch, shuffle=True,
                                     use_select=use_select, select_params=select_params,
                                     weak_label_dir=opt.data.weak_label_dir)
                ds_val = BGDataset(data_dir, use_weak=opt.data.use_weak, phase='val',
                                   batch_size=opt.train.valid_batch, shuffle=False,
                                   use_select=use_select, select_params=select_params,
                                   weak_label_dir=opt.data.weak_label_dir)
                self.val_ids = ds_val.samples
                patch_size = (next(ds_val))['data'].shape[-3:]
                # warmup and go through the dataset
                # sum(1 for _ in ds_train); sum(1 for _ in ds_val)
                # print('length: ds_train--%02d, ds_val--%02d' % (sum(1 for _ in ds_train), sum(1 for _ in ds_val)))

                # dataloader params
                extra_label_keys = ['weak_label'] if opt.data.use_weak else None
                if opt.data.weak_label_dir or self.EM_save:
                    extra_label_keys = ['weak_label', 'pseudo_label']
                if opt.sp.use_type != '':
                    extra_label_keys = extra_label_keys + ['sp_edge'] if extra_label_keys is not None \
                        else ['sp_edge']    # 'sp_region'


                if opt.model.use_author_block:
                    pool_op_kernel_sizes = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
                else:
                    pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                    np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size, \
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   seeds_train=args.seed, seeds_val=args.seed,
                                                                   pin_memory=True,
                                                                   anisotropy=opt.data.anisotropy,
                                                                   extra_label_keys=extra_label_keys)
                data_loaders = {
                    'train': train_loader,
                    'val': val_loader,
                }
                # print('Num per train_loader', sum(1 for _ in data_loaders['train']))
                # print('Num per valid_loader', sum(1 for _ in data_loaders['val']))

                # load extra dataloader for saving pseudo labels
                if self.EM_save:
                    extra_label_keys = ['weak_label'] if opt.data.use_weak else None
                    ds_train = BGDataset(opt.data.data_dir, use_weak=opt.data.use_weak, phase='train',
                                         batch_size=opt.train.valid_batch, shuffle=False,
                                         use_conf=opt.data.use_conf)
                    ds_val = BGDataset(opt.data.data_dir, use_weak=opt.data.use_weak, phase='val',
                                       batch_size=opt.train.valid_batch, shuffle=False)
                    self.save_val_ids = {'train': ds_train.samples, 'val': ds_val.samples}
                    # dataloader params
                    val_mode = True
                    self.E_train_loader, _ = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size, \
                                                                       deep_supervision_scales=deep_supervision_scales,
                                                                       pin_memory=True,
                                                                       anisotropy=opt.data.anisotropy,
                                                                       extra_label_keys=extra_label_keys,
                                                                       val_mode=val_mode)

            elif opt.data.dataset == 'AEDataset':
                # data dir
                train_dir, val_dir = opt.data.train_dir, opt.data.val_dir

                # dataset
                noise_params = {
                    'extra_near_nums': opt.ae.extra_near_nums,
                    'extra_radius':    opt.ae.extra_radius,
                    'extra_slices':    opt.ae.extra_slices,
                    'dilate_nums': opt.ae.dilate_nums,      # dilated regions near pred boundaries
                    'dilate_iters': opt.ae.dilate_iters,    # dilated iters(size) for each new region
                    'extend_nums_head': opt.ae.extend_nums_head,      # elongation along trachea head
                    'extend_nums_branch': opt.ae.extend_nums_branch,  # elongation along trachea tail
                    'close_iters': opt.ae.close_iters,      # closing operation iters for the whole region
                }
                if opt.data.anisotropy:
                    noise_params['erode_nums'] = opt.ae.erode_nums
                    noise_params['erode_iters'] = opt.ae.erode_iters
                    noise_params['erode_slices'] = opt.ae.erode_slices


                ds_train = AEDataset(train_dir, val_dir, use_weak=opt.data.use_weak, phase='train',
                                     batch_size=opt.train.train_batch, shuffle=True,
                                     noise_params=noise_params,
                                     use_pred_num= opt.ae.use_pred_num, use_prob_top=opt.ae.use_prob_top,
                                     pseudo_label_lcc= opt.ae.pseudo_label_lcc, label_percent=opt.ae.label_percent,
                                     anisotropy=opt.data.anisotropy)
                ds_val = AEDataset(train_dir, val_dir, use_weak=opt.data.use_weak, phase='val',
                                   batch_size=opt.train.valid_batch, shuffle=False,
                                   anisotropy=opt.data.anisotropy)
                self.val_ids = ds_val.samples
                patch_size = (next(ds_val))['data'].shape[-3:]

                # dataloader params
                global_params = {
                    "do_scaling": opt.ae.do_scaling, "p_scale": opt.ae.p_scale, "do_rotation": opt.ae.do_rotation,
                    "p_rot": opt.ae.p_rot, "do_translate": opt.ae.do_translate, "p_trans": opt.ae.p_trans,
                    "trans_max_shifts": {'z': opt.ae.trans_max_shifts, 'y': opt.ae.trans_max_shifts, 'x': opt.ae.trans_max_shifts},
                }

                extra_label_keys = ['pred']
                if opt.model.use_author_block:
                    pool_op_kernel_sizes = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
                else:
                    pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                    np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                train_loader, val_loader = get_moreDA_augmentation_ae(ds_train, ds_val, patch_size=patch_size, \
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   seeds_train=args.seed, seeds_val=args.seed,
                                                                   pin_memory=True,
                                                                   anisotropy=opt.data.anisotropy,
                                                                   extra_label_keys=extra_label_keys,
                                                                   global_params=global_params)

                data_loaders = {
                    'train': train_loader,
                    'val': val_loader,
                }



        else:   # only val
            # self.phases = ['val']
            if ',' in args.demo:
                [self.phases.append(phase.strip()) for phase in (args.demo).split(',')]
                print(self.phases)
            else:
                self.phases = [(args.demo).strip()]

            if opt.data.dataset == 'BGDataset':
                ds_train = BGDataset(opt.data.data_dir, use_weak=opt.data.use_weak, phase='train',
                                     batch_size=opt.train.valid_batch, shuffle=False,
                                     use_conf=opt.data.use_conf)
                ds_val = BGDataset(opt.data.data_dir, use_weak=opt.data.use_weak, phase='val',
                                   batch_size=opt.train.valid_batch, shuffle=False)
                self.val_ids = {'train': ds_train.samples, 'val': ds_val.samples}

                if 'test' in self.phases:
                    ds_test = BGDataset(opt.data.data_dir, use_weak=opt.data.use_weak, phase='test',
                                   batch_size=opt.train.valid_batch, shuffle=False)
                    self.val_ids['test'] = ds_test.samples
                    patch_size = (next(ds_test))['data'].shape[-3:]
                else:
                    patch_size = (next(ds_val))['data'].shape[-3:]
                # dataloader params
                extra_label_keys = ['weak_label'] if opt.data.use_weak else None
                if opt.sp.use_type != '':
                    extra_label_keys = extra_label_keys + ['sp_edge'] if extra_label_keys is not None \
                        else ['sp_edge']    # 'sp_region'
                if opt.model.use_author_block:
                    pool_op_kernel_sizes = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
                else:
                    pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)

                if not opt.data.use_conf:
                    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                        np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                    val_mode = True
                    use_conf = False
                else:
                    deep_supervision_scales = [[1, 1, 1]]
                    val_mode = False
                    use_conf = True

                train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size, \
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   pin_memory=True,
                                                                   anisotropy=opt.data.anisotropy,
                                                                   extra_label_keys=extra_label_keys,
                                                                   val_mode=val_mode,
                                                                   use_conf=use_conf)    #
                if 'test' in self.phases:
                    _, test_loader =  get_moreDA_augmentation(None, ds_test, patch_size=patch_size, \
                                                                       deep_supervision_scales=deep_supervision_scales,
                                                                       pin_memory=True,
                                                                       anisotropy=opt.data.anisotropy,
                                                                       extra_label_keys=extra_label_keys,
                                                                       val_mode=val_mode,
                                                                       use_conf=use_conf)

            elif opt.data.dataset == 'AEDataset':
                # data dir
                train_dir, val_dir = opt.data.train_dir, opt.data.val_dir

                # dataset
                ds_train = AEDataset(train_dir, val_dir, use_weak=opt.data.use_weak, phase='train',
                                     batch_size=opt.train.valid_batch, shuffle=False, val_mode=True, use_train_all=True,
                                     use_pred_num=opt.ae.use_pred_num, use_prob_top=opt.ae.use_prob_top,
                                     pseudo_label_lcc=opt.ae.pseudo_label_lcc, label_percent=opt.ae.label_percent,
                                     anisotropy=opt.data.anisotropy)
                ds_val = AEDataset(train_dir, val_dir, use_weak=opt.data.use_weak, phase='val',
                                   batch_size=opt.train.valid_batch, shuffle=False,
                                   anisotropy=opt.data.anisotropy)
                self.val_ids = {'train': ds_train.samples, 'val': ds_val.samples}
                patch_size = (next(ds_val))['data'].shape[-3:]
                # dataloader params
                extra_label_keys = ['pred']
                pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                        np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                val_mode = True
                train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size, \
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   seeds_train=args.seed, seeds_val=args.seed,
                                                                   pin_memory=True,
                                                                   anisotropy=opt.data.anisotropy,
                                                                   extra_label_keys=extra_label_keys,
                                                                   val_mode=val_mode)

            data_loaders = {
                'train': train_loader,
                'val': val_loader,
            }
            if 'test' in self.phases:
                data_loaders['test'] = test_loader

        return data_loaders, None, None

    def build_model(self):
        if opt.model.network == 'ResidualUNet':
            model = eval(opt.model.network)(in_dim=1, out_dim=2)
        elif opt.model.network == 'VNet':
            model = eval(opt.model.network)()
        elif opt.model.network == 'Generic_UNet':
            if opt.model.use_author_block:
                self.pool_op_kernel_sizes = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
                conv_kernel_sizes = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
                # print('Use the same blocks as Authors of nn-UNet')
            else:
                self.pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                conv_kernel_sizes = ast.literal_eval(opt.model.conv_kernel_sizes)

            input_channel = opt.model.model1_fchannel if (opt.model.use_model1 and opt.model.model1_fchannel>0) else 1
            if opt.ae.input == 'pred_aug' and opt.ae.input_image_type == 'concat':
                input_channel = 2

            net_params = {
                'input_channels': input_channel, 'base_num_features': 32, 'num_classes': 2,
                'num_pool': len(self.pool_op_kernel_sizes), 'disable_skip': opt.model.disable_skip,

                'num_conv_per_stage': 2, 'feat_map_mul_on_downscale': 2, 'conv_op': nn.Conv3d,
                'norm_op': nn.InstanceNorm3d, 'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': nn.Dropout3d, 'dropout_op_kwargs': {'p': 0, 'inplace': True},
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
                'deep_supervision': opt.model.deep_supervision, 'dropout_in_localization': False,
                'final_nonlin': lambda x: x,

                'pool_op_kernel_sizes': self.pool_op_kernel_sizes,
                'conv_kernel_sizes': conv_kernel_sizes,
                'upscale_logits': False, 'convolutional_pooling': True, 'convolutional_upsampling': True,
                'gnn_type': opt.gnn.gnn_type, 'n_blocks': opt.gnn.n_blocks, 'resolution': opt.gnn.resolution,
                'sub_sample': opt.gnn.sub_sample, 'bn_layer': opt.gnn.bn_layer,
                'sp_use_type': opt.sp.use_type
            }
            if opt.ae.input == 'pred':
                model = AutoEnc(**net_params)
            else:
                model = eval(opt.model.network)(**net_params)
            if opt.model.use_model1:
                net_params['input_channels'] = 1
                if opt.model.model1_fchannel > 0:
                    net_params['return_feature'] = True
                self.model1 = eval(opt.model.network)(**net_params)

        if args.demo != '': # load pretrained model
            # print('=' * 40, 'Loading pretrained model to test...')
            if args.parallel:
                model = nn.DataParallel(model)
            model_dict = torch.load(args.weight_path)
            model.load_state_dict(model_dict)
        elif opt.model.use_finetune and opt.model.finetune_model_path!='':
            # print('=' * 40, 'Loading pretrained model to model...')
            if args.parallel:
                model = nn.DataParallel(model)
            model_dict = torch.load(opt.model.finetune_model_path)
            model.load_state_dict(model_dict)

        elif opt.model.use_model1 and opt.model.model1_path != '':
            # print('=' * 40, 'Loading pretrained model to model1...')
            if args.parallel:
                self.model1 = nn.DataParallel(self.model1)
                model = nn.DataParallel(model)
            model_dict = torch.load(opt.model.model1_path)
            self.model1.load_state_dict(model_dict)
            self.model1 = self.model1.cuda()

        elif opt.ae.input == 'pred':
            init_model_dict = model.state_dict()
            pre_model_dict = torch.load(args.weight_path)
            for key in pre_model_dict.keys():
                # print(key)
                assert 'pred_module.{}'.format(key[7:]) in init_model_dict.keys()
                init_model_dict['pred_module.{}'.format(key[7:])] = copy.deepcopy(pre_model_dict[key])
            model.load_state_dict(init_model_dict)
            
            model_params = model.named_parameters()
            for name, param in model_params:
                prefix = name.split('.')[0]
                # print(prefix, name)
                if prefix in ['pred_module']:
                    param.requires_grad = False
                    # print('     ', prefix, name)

            if args.parallel:
                model = nn.DataParallel(model)
        elif opt.sp.finetune != '':
            if args.parallel:
                model = nn.DataParallel(model)
            init_model_dict = model.state_dict()
            pre_model_dict = torch.load(args.weight_path)
            for key in pre_model_dict.keys():
                init_model_dict[key] = copy.deepcopy(pre_model_dict[key])
            model.load_state_dict(init_model_dict)
        else:
            if args.parallel:
                model = nn.DataParallel(model)

        # load extra autoencoder model
        if opt.model.use_emiter and (opt.model.filter_phase != 'prob'):
            ae_model = copy.deepcopy(model)
            ae_model.load_state_dict(torch.load(opt.model.finetune_model_path_ae))
            self.ae_model = ae_model.cuda()
        # ae post_process
        if (args.demo != '') and (args.ae_weight_path != ''):
            ae_model = copy.deepcopy(model)
            ae_model.load_state_dict(torch.load(args.ae_weight_path))
            self.ae_model = ae_model.cuda()

        model = model.cuda()

        return model


    def load_model_state_dict(self, model):
        model_dir = os.path.join(opt.data_dir, opt.vgg)
        pre_model_dict = torch.load(model_dir)
        init_model_dict = model.state_dict()
        count = 0
        for key in pre_model_dict.keys():
            if 'features' in key:
                init_key = list(init_model_dict.keys())[count]
                pre_key = list(pre_model_dict.keys())[count]
                init_model_dict[init_key] = copy.deepcopy(pre_model_dict[pre_key])
                count += 1
                # logging.info(init_key, pre_key)
        model.load_state_dict(init_model_dict)
        return model


    def build_optimizer(self):
        class_weights = torch.Tensor([1.0, opt.train.fg_weight]).cuda() # only for weak now
        if opt.train.loss_name == 'MultipleOutputLoss2':
            # loss weights
            net_numpool = len(self.pool_op_kernel_sizes)
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            # print('loss weights:\t', weights)

            # Fully supervision: ce_loss + dice_loss
            # Weak supervision: ce_loss
            if (not opt.data.use_weak) or opt.train.graphcut or opt.ae.input is not None:
                multi_loss = False  # not implemented
                batch_dice = True
                loss_func = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
                ce_loss = False
            else:
                multi_loss = opt.train.draw_fgbg
                if opt.train.focal_loss:
                    loss_func = FocalLoss(weight=class_weights, ignore_index=255)
                elif opt.train.with_dice_loss:
                    loss_func = CEDiceLoss(weight=class_weights, ignore_index=255)
                else:
                    loss_func = CELoss(weight=class_weights, ignore_index=255)
                ce_loss = True
            criterion = eval(opt.train.loss_name)(loss_func, weights, ce_loss=ce_loss, multi_loss=multi_loss)

            # independent pseudo loss: for weak+pseudo supervised setting
            if opt.train.ind_pseudo_loss != '':
                if opt.train.ind_pseudo_loss == 'focal':
                    pseudo_loss_func = CELoss(weight=class_weights, ignore_index=255)
                elif opt.train.ind_pseudo_loss == 'ce':
                    pseudo_loss_func = CELoss(weight=class_weights, ignore_index=255)
                self.pseudo_criterion = eval(opt.train.loss_name)(pseudo_loss_func, weights, ce_loss=ce_loss, multi_loss=multi_loss)
            else:
                self.pseudo_criterion = criterion

            if opt.sp.loss != '':
                loss_func = RegLoss(loss_func=opt.sp.loss, ignore_index=255)
                self.sp_criterion = eval(opt.train.loss_name)(loss_func, weights, ce_loss=ce_loss, multi_loss=multi_loss, regress=True)
            else:
                self.sp_criterion = criterion
        else:
            criterion = eval(opt.train.loss_name)(weight=class_weights)
        metric = DiceMetric(dice_each_class=False)   # fg & bg dice
        optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.train.lr,
                                    momentum=opt.train.momentum, weight_decay=opt.train.weight_decay)
        if opt.train.lr_decay == 'multistep':
            milestones = ast.literal_eval(opt.train.milestones)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=opt.train.gamma)
        elif opt.train.lr_decay == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.train.plateau_patience, factor=opt.train.plateau_gamma)
        else:   # {constant, poly}
            scheduler = None

        return criterion, metric, optimizer, scheduler

    def update_optimizer(self):
        if opt.train.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.train.lr_update,
                                             weight_decay=opt.train.weight_decay)
        elif opt.train.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.train.lr_update,
                                         momentum=opt.train.momentum, weight_decay=opt.train.weight_decay)
        else:
            raise NotImplementedError
            # print('Invalid Optimizer: %s' % (opt.train.optimizer))

    def get_input_for_auto_encoder(self, targets):
        inputs = copy.deepcopy(targets[0]).float()  # test auto encoder
        if opt.ae.keep_slices < 1000:
            tmp_inputs = copy.deepcopy(inputs) * 0.
            for i in range(inputs.shape[0]):
                nonzero = torch.nonzero(inputs[i, 0])
                z0, z1 = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
                num_slices = z1 - z0 + 1
                if num_slices <= opt.ae.keep_slices:
                    tmp_inputs[i] = copy.deepcopy(inputs[i])
                else:
                    keep_ids = random.sample(range(z0, z1 + 1), min(opt.ae.keep_slices, num_slices))
                    for j in keep_ids:
                        tmp_inputs[i, 0, j] = copy.deepcopy(inputs[i, 0, j])
            inputs = tmp_inputs
        return inputs

    def train(self):
        if opt.sp.use_type != '':
            self.train_superpixel()
        else:
            self.train_before_0818()
    
    def train_before_0818(self):
        num_epochs = opt.train.n_epochs
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        final_dice_meter = MultiLossMeter()     # for val dice after ae processing
        total_iter = {'train': 0, 'val': 0, 'test': 0}
        # warmup dataloader
        if not args.save_em_init:   # default setting
            print('Num per train_loader', sum(1 for _ in self.data_loaders['train']))
            print('Num per valid_loader', sum(1 for _ in self.data_loaders['val']))

        # extra init params
        last_fpr = -1

        for epoch in tqdm(range(num_epochs)):

            # Hyperparams: variable flag_fpr_drop: fp_filter_ratio drops 0.1 in curriculum learning
            if opt.model.use_emiter:
                if last_fpr == -1:  # init some params
                    fp_filter_ratio, fn_filter_ratio = opt.model.fp_filter_ratio, opt.model.fn_filter_ratio
                    filter_k_times = opt.model.filter_k_times
                    last_fpr = opt.model.fp_filter_ratio
                if (opt.model.fpr_drop_freq > 0) and (epoch > 0) and (epoch % opt.model.fpr_drop_freq == 0):    #
                    if last_fpr > opt.model.fpr_min_ratio:  # fp_filter_ratio drops 0.1
                        logging.info('FPR drops: From %.2f to %.2f!' % (last_fpr, last_fpr - opt.model.fpr_drop_rate))
                        fp_filter_ratio = last_fpr - opt.model.fpr_drop_rate
                        last_fpr = last_fpr - opt.model.fpr_drop_rate
                        # load best model of past epochs
                        model_dict = torch.load('{}/best_model.pth'.format(opt.model_dir))
                        self.model.load_state_dict(model_dict)
                        logging.info('Loaded Past Best Model!')

                if args.save_em_init or ((opt.model.iterate_epoch > 0) and (epoch % opt.model.iterate_epoch == 0))\
                        or (opt.model.filter_extra == 'modify_initial_weak' and epoch == 0):
                    # inference train set, save to a fix dir, for later dataloader.
                    if epoch > 0:
                        # load best model of past epochs
                        model_dict = torch.load('{}/best_model.pth'.format(opt.model_dir))
                        self.model.load_state_dict(model_dict)
                        logging.info('Loaded Past Best Model!')
                    self.E_model_generate_pseudo(fp_filter_ratio, fn_filter_ratio, filter_k_times)

            for phase in self.phases:
                if phase == 'train' or phase == 'train_val':
                    self.model.train(True)
                    ## test bootstrap performance on val set.
                    # if epoch == 0:
                    #     continue
                else:
                    final_dice_meter.reset()
                    self.model.train(False)
                    sum(1 for _ in self.data_loaders['val'])    # for debugging val_loader error: unexpected number
                loss_meter.reset(); dice_meter.reset()
                count_batch = 0

                while count_batch < opt.train.n_batches:
                    for i_batch, data in enumerate(self.data_loaders[phase]):
                        if opt.train.loss_name == 'MultipleOutputLoss2':
                            if opt.ae.input is None:
                                data_image = data['image'].float().cuda()
                                if opt.data.use_weak:
                                     data_weak_label = [it.long().cuda() for it in data['weak_label']]
                            data_gt = [it.long().cuda() for it in data['gt']]

                            # 0. target supervision
                            if opt.ae.input is not None:
                                targets = data_gt    # processed pseudo label in fact
                            else:
                                targets = data_weak_label if opt.data.use_weak else data_gt

                            # 0.1 extra target supervision
                            targets_pseudo = None
                            if opt.data.use_select or opt.data.weak_label_dir != '':
                                # A. pseudo label from generated preds
                                if opt.model.use_model1:        # below to be discarded
                                    # generated preds
                                    inputs = data_image
                                    with torch.no_grad():
                                        if opt.model.model1_fchannel > 0:
                                            Features, F_outputs = self.model1(inputs)  # Features: (B,64, ...)
                                        else:
                                            F_outputs = self.model1(inputs)
                                        F_preds = [nn.Softmax(dim=1)(F_output)[:, 1:, ...] for F_output in F_outputs]
                                        # select samples with pseudo label
                                        tag_sum = data['pseudo_tag'].sum()
                                        if tag_sum == inputs.shape[0]:  # all selected
                                            targets_pseudo = F_preds
                                        elif tag_sum == 0:              # no selected
                                            targets_pseudo = None
                                        else:
                                            targets_pseudo = [torch.zeros_like(plabel) for plabel in F_preds]
                                            tlength = len(targets_pseudo)
                                            for kth, tag in enumerate(data['pseudo_tag']):
                                                if tag == 1:
                                                    for jth in range(tlength):
                                                        targets_pseudo[jth][kth:kth+1] = F_preds[jth][kth:kth+1]
                                else:
                                    targets_pseudo = data['pseudo_label']
                                if targets_pseudo is not None:
                                    targets_pseudo = [it.long().cuda() for it in targets_pseudo]  # deep supervision
                            # 0.1.2 EMIter: pseudo label for em in each iteration
                            elif opt.model.use_emiter:
                                if opt.model.iterate_epoch <= 0:    # EM alternates per iteration: generate pseudo label
                                    weak_label = data_weak_label[0]
                                    inputs = data_image
                                    with torch.no_grad():   # E step forward
                                        outputs = self.model(inputs, opt.model.feat_for_pseudo)
                                        if opt.model.feat_for_pseudo is not None:
                                            features, outputs = outputs
                                        logits = nn.Softmax(dim=1)(outputs[0])
                                        probs = logits[:, 1, ...]    # last prob volume

                                    if opt.model.pseudo_wce != '':
                                        em_wce_phase = opt.model.pseudo_wce
                                        pseudos, pseudos_weights = self.filter2pseudo_weights(logits, wce_phase=em_wce_phase, weak_labels=weak_label, gt=data_gt[0])   #
                                    else:
                                        filter_phase = opt.model.filter_phase   # {'prob', 'ae', 'sdf', 'sdfbg', 'probae', 'probsdf'}
                                        filter_extra_mode = opt.model.filter_extra
                                        pseudos = self.filter2pseudo(probs, logits=outputs[0], filter_phase=filter_phase, weak_labels=weak_label, extra_mode=filter_extra_mode,
                                                fp_filter_ratio=fp_filter_ratio, fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times)
                                        if opt.model.feat_for_pseudo is not None:
                                            pseudos = self.update_pseudo_by_feature(pseudos, features)
                                        if 'modify_weak_label' in filter_extra_mode:
                                            targets[0][(pseudos.unsqueeze(1) != 1).long() + (targets[0] == 1).long() == 2] = 255


                                    if opt.train.pseudo_one_supervision:
                                        out_shapes = [outputs[0].shape]
                                    else:
                                        out_shapes = [output.shape for output in outputs]
                                    targets_pseudo = downsample_seg_scales(pseudos, out_shapes)
                                else:       # EM alternates per some epoch: load pseudo label
                                    targets_pseudo = data['pseudo_label']
                                targets_pseudo = [it.long().cuda() for it in targets_pseudo]

                            # 1. input
                            if opt.ae.input == 'gt':
                                inputs = self.get_input_for_auto_encoder(targets)
                            elif opt.ae.input == 'pred_aug':
                                inputs = data['pred'][0].float().cuda()
                                if opt.ae.input_image_type == 'add':     # input: image+mask
                                    inputs = inputs + data['image'].cuda()
                                elif opt.ae.input_image_type == 'concat':     # input: [image|mask]
                                    inputs = torch.cat((inputs, data_image), 1)
                            elif opt.model.use_model1:
                                if opt.model.model1_fchannel > 0:
                                    inputs = Features
                                else:
                                    inputs = data_image
                            else:
                                inputs = data_image

                            # 2. forward
                            outputs = self.model(inputs)
                            preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]  # preds (B, H, W)

                        else:   # vanilla loss: current no use
                            inputs = data['image'].float().cuda()
                            targets = data['weak_label'] if opt.data.use_weak else data['gt']
                            targets = targets.long().cuda()
                            outputs = self.model(inputs)
                            preds = nn.Softmax(dim=1)(outputs)[:, 1, ...]  # preds (B, H, W)

                        # 3. loss: train mode, val mode
                        if (phase == 'train' or phase == 'train_val'):
                            losses, loss_names = self.criterion(outputs, targets)
                            if targets_pseudo is not None:  # add pseudo label supervision
                                if opt.data.select_use_pred_num == 1000 or opt.model.use_emiter:    # all with pseudo label
                                    disable_auto_weight = True if not opt.train.pseudo_auto_weight else False
                                    if not opt.train.pseudo_one_supervision:    # deep supervision
                                        losses_pseudo_sum, _ = self.pseudo_criterion(outputs, targets_pseudo)
                                    else:                                       # only one supervision
                                        losses_pseudo_sum = [0.0 for _ in losses]
                                        if opt.model.pseudo_wce != '':
                                            losses_pseudo_sum[:2], _ = self.pseudo_criterion(outputs[:1], targets_pseudo[ :1], pixel_weights=pseudos_weights, disable_auto_weight=disable_auto_weight)  # loss, d0
                                        else:
                                            losses_pseudo_sum[:2], _ = self.pseudo_criterion(outputs[:1], targets_pseudo[:1], disable_auto_weight=disable_auto_weight)  # loss, d0
                                else:   # one by one
                                    losses_pseudo_sum = [0.0 for _ in losses]; real_pseudo_num = 0
                                    for batch_idx in range(targets_pseudo[0].shape[0]):
                                        current_pseudo = [pseudo[batch_idx: batch_idx+1] for pseudo in targets_pseudo]
                                        if 1 in current_pseudo[0]:  # real pseudo label
                                            current_output = [output[batch_idx: batch_idx+1] for output in outputs]
                                            losses_pseudo, _ = self.pseudo_criterion(current_output, current_pseudo)
                                            losses_pseudo_sum = [losses_pseudo_sum[kth]+losses_pseudo[kth] for kth in range(len(losses_pseudo))]
                                            real_pseudo_num += 1
                                    if losses_pseudo_sum[0] != 0.0:
                                        losses_pseudo_sum = [term_sum / real_pseudo_num for term_sum in losses_pseudo_sum]

                                # log record two separate losses: weak, pseudo
                                if opt.train.log_two_loss:
                                    log_record_losses = {'pseudo_loss': losses_pseudo_sum[0], 'weak_loss': losses[0]}
                                if opt.train.only_pseudo_loss:
                                    losses = losses_pseudo_sum
                                else:   # weak + pseudo losses
                                    pweight = opt.train.pseudo_loss_weight  # default 1
                                    wweight = opt.train.weak_loss_weight    # default 1
                                    # # debug
                                    # print('weak_loss:', losses[0] * wweight, 'pseudo_loss:', losses_pseudo_sum[0] * pweight)
                                    losses = [losses[kth] * wweight + losses_pseudo_sum[kth] * pweight for kth in range(len(losses))]

                            loss = losses[0]
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        else:
                            with torch.no_grad():
                                losses, loss_names = self.criterion(outputs, targets)
                            loss = losses[0]
                        loss_meter.update(losses, loss_names)

                        # ipdb.set_trace()
                        # print(inputs[0, 0, 0, 0, 0], outputs[0][0, 0, 0, 0, 0], loss)
                        # print(inputs[1, 0, 0, 0, 1], outputs[0][1, 0, 0, 0, 1], loss)

                        # extract real gt for weak setting
                        gt = data_gt
                        if opt.train.loss_name == 'MultipleOutputLoss2':
                            dices, dice_names = self.metric.forward(preds, gt[0])
                        else:
                            dices, dice_names = self.metric.forward(preds, gt)
                        dice_meter.update(dices, dice_names)

                        #TODO val: AE process to get final preds, for later best val dice
                        if phase == 'val' and opt.model.use_emiter and opt.model.filter_phase in ['probae']:
                            with torch.no_grad():
                                # ae_inputs = torch.unsqueeze(preds.float(), dim=1) # error
                                ae_inputs = torch.unsqueeze((preds > 0.5).float(), dim=1)
                                ae_logits = self.ae_model(ae_inputs)[0]
                                ae_probs = nn.Softmax(dim=1)(ae_logits)[:, 1, ...]
                                ae_preds = ae_probs
                                # ae_preds = (ae_probs > 0.5).long()  # (B, D, H, W) # no use
                            final_val_dices, final_dice_names = self.metric.forward(ae_preds, gt[0])
                            final_dice_names = [it+'_final' for it in final_dice_names]
                            final_dice_meter.update(final_val_dices, final_dice_names)

                        # plot running loss curve
                        if opt.data.write_log_batch:    # logging per iteration
                            metrics = {'loss': loss, 'dice': dices[0]}  #\'dice_bg': dices[1]
                            # self.plot_curves(metrics, None, total_iter[phase], phase=phase)
                            total_iter[phase] += 1
                            if phase=='val' or (not opt.train.log_two_loss):  # default
                                self.log_batch(metrics, epoch, i_batch)
                            else:
                                metrics['pseudo_loss'] = log_record_losses['pseudo_loss']
                                metrics['weak_loss'] = log_record_losses['weak_loss']
                                self.log_batch_two_loss(metrics, epoch, i_batch)
                                del log_record_losses

                        # plot multiple loss curve
                        # loss_terms = dict(zip(loss_names, losses))
                        # self.plot_curves_multi(loss_terms, None, total_iter[phase], phase=phase)

                        count_batch += 1
                        if count_batch >= opt.train.n_batches:
                            break

                        # release memory
                        del inputs, targets, gt, preds, outputs, losses, dices
                        if phase == 'val' and opt.model.use_emiter and opt.model.filter_phase in ['probae']: # ae process related params
                            del ae_inputs, ae_logits, ae_probs, ae_preds

                    if phase == 'val':
                        break

                avg_terms = loss_meter.get_metric()
                dice_terms = dice_meter.get_metric()
                avg_terms = {**avg_terms, **dice_terms}
                best_dice_name = 'dice'; extra_key = None
                #TODO val: add final val dice
                if phase == 'val' and opt.model.use_emiter and opt.model.filter_phase in ['probae']:
                    final_terms = final_dice_meter.get_metric()
                    avg_terms = {**avg_terms, **final_terms}
                    best_dice_name = best_dice_name + '_final'
                    extra_key = best_dice_name

                self.plot_curves_multi(avg_terms, epoch, phase=phase)

                if phase == 'val' or phase == 'train_val':
                    if avg_terms[best_dice_name] > self.best[best_dice_name]:
                        self.update_best(avg_terms, epoch, extra_key=extra_key)
                self.log_epoch(avg_terms, epoch, phase, extra_key=extra_key)

            # # release memory
            # del inputs, targets, gt, preds, outputs, losses, dices

            if epoch % opt.snapshot == 0:
                snapshot = copy.deepcopy(self.model)
                torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.model_dir, epoch))

            # adjust lr on epoch level
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_terms['dice'])
                else:
                    self.scheduler.step()
            elif opt.train.lr_decay == 'poly':
                lr_ = opt.train.lr * (1 - epoch / opt.train.n_epochs) ** 0.9  # power=0.9
                if lr_ < opt.train.min_lr:  # minimum lr
                    lr_ = opt.train.min_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

        self.log_final()
        self.writer.close()

    def filter2pseudo_weights(self, logits, wce_phase='', weak_labels=None, gt=None):
        '''
        :return: pseudo label, pixel-wise loss weight for WCE
        ** gt only for debug
        '''
        probs = logits[:, 1, ...]
        if wce_phase == 'prob_wce':
            pseudos, pseudos_weights = label_weigh_uncertain(probs, phase=wce_phase, weak_labels=weak_labels,
                     method=opt.model.pseudo_wce_func, gamma=opt.model.pseudo_wce_powk, mu=opt.model.pseudo_wce_mu)
        elif wce_phase == 'ae_wce':
            with torch.no_grad():
                ae_inputs = torch.unsqueeze((probs > 0.5).float(), dim=1)
                ae_outputs = self.ae_model(ae_inputs)
                ae_probs = nn.Softmax(dim=1)(ae_outputs[0])[:, 1, ...]
                ae_preds = (ae_probs > 0.5).long()  # (B, D, H, W)
            pseudos, pseudos_weights = label_weigh_uncertain(probs, ae_preds=ae_preds, phase=wce_phase,
                     weak_labels=weak_labels, method=opt.model.pseudo_wce_func, gamma=opt.model.pseudo_wce_powk,
                     mu=opt.model.pseudo_wce_mu)
        '''
        # debug: save gt, pred, pseudo, pseudo_weight
        save_dir = '/group/lishl/weak_exp/tmp/'
        for ith in range(pseudos.shape[0]):
            gt_np, pred_np, pseudo_np, pweight_np = gt[ith, 0].cpu().numpy(), ((probs > 0.5).long())[ ith].cpu().numpy(), \
                                                    pseudos[ith].cpu().numpy(), pseudos_weights[ith].cpu().numpy()
            nib.save(nib.Nifti1Image(gt_np.astype(np.uint8), np.eye(4)), save_dir + 'gt_%d.nii.gz' % ith)
            nib.save(nib.Nifti1Image(pred_np.astype(np.uint8), np.eye(4)), save_dir + 'pred_%d.nii.gz' % ith)
            nib.save(nib.Nifti1Image(pseudo_np.astype(np.uint8), np.eye(4)), save_dir + 'pseudo_%d.nii.gz' % ith)
            nib.save(nib.Nifti1Image(pweight_np, np.eye(4)), save_dir + 'pweight_%d.nii.gz' % ith)
        ipdb.set_trace()
        '''

        return pseudos, pseudos_weights

    def filter2pseudo(self, probs, logits=None, filter_phase='prob', weak_labels=None,
                      fp_filter_ratio=0, fn_filter_ratio=0, filter_k_times=0,
                      extra_mode=''):
        if filter_phase == 'prob':
            pseudos = label_filter_uncertain(probs, phase=filter_phase, weak_labels=weak_labels,
                      fp_filter_ratio=fp_filter_ratio, fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times, extra_mode=extra_mode)
        else:
            with torch.no_grad():
                ae_inputs = torch.unsqueeze((probs > 0.5).float(), dim=1)
                ae_logits = self.ae_model(ae_inputs)[0]
                ae_probs = nn.Softmax(dim=1)(ae_logits)[:, 1, ...]
                ae_preds = (ae_probs > 0.5).long()  # (B, D, H, W)
            if filter_phase in ['ae']:
                pseudos = label_filter_uncertain(ae_probs, phase=filter_phase,
                                                 weak_labels=weak_labels, fp_filter_ratio=fp_filter_ratio,
                                                 fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times, extra_mode=extra_mode)
            elif filter_phase in ['ae_psrank', 'sdf', 'sdfbg']:
                sdfs = calculate_sdf(ae_preds)
                pseudos = label_filter_uncertain(ae_probs, phase=filter_phase, sdfs=sdfs, psrk=opt.model.psrk,
                                                 weak_labels=weak_labels, fp_filter_ratio=fp_filter_ratio,
                                                 fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times, extra_mode=extra_mode)
            elif filter_phase in ['probae', 'aeprob', 'probae_addrank', 'probae_adaptive']:
                pseudos = label_filter_uncertain(probs, phase=filter_phase, ae_labels=ae_preds, ae_probs=ae_probs,
                                                 weak_labels=weak_labels, fp_filter_ratio=fp_filter_ratio,
                                                 fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times, extra_mode=extra_mode)
            elif filter_phase == 'probsdf':
                sdfs = calculate_sdf(ae_preds)
                pseudos = label_filter_uncertain(probs, phase=filter_phase, sdfs=sdfs,
                                                 weak_labels=weak_labels, fp_filter_ratio=fp_filter_ratio,
                                                 fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times, extra_mode=extra_mode)
            elif filter_phase in ['t_scaling']:
                temperatures = [opt.model.t_scaling_t_prob, opt.model.t_scaling_t_ae]
                combine_mode = opt.model.t_scaling_combine
                pseudos = label_filter_uncertain(probs, phase=filter_phase, weak_logits=logits, ae_logits=ae_logits, ae_labels=ae_preds,
                                                 t_scaling_temperatures=temperatures, t_scaling_combine=combine_mode,
                                                 weak_labels=weak_labels, fp_filter_ratio=fp_filter_ratio, fn_filter_ratio=fn_filter_ratio,
                                                 filter_k_times=filter_k_times, extra_mode=extra_mode)

        return pseudos

    def update_best(self, data, epoch, extra_key=None):
        self.best = {'loss': data['loss'], 'dice': data['dice'], 'epoch': epoch}
        if extra_key is not None:
            self.best[extra_key] = data[extra_key]
        best_model = copy.deepcopy(self.model)
        torch.save(best_model.cpu().state_dict(), '{}/best_model.pth'.format(opt.model_dir))
        logging.info('=' * 40 + 'update best model')

    def weights_init2(self, m):
        """ Weight initialization function. """
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                # Constant initialization for fusion layer in HED network.
                torch.nn.init.constant_(m.weight, 0.2)
            else:
                # Zero initialization following official repository.
                # Reference: hed/docs/tutorial/layers.md
                m.weight.data.zero_()
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()

    def plot_curves(self, data, epoch, iter = -1, phase='train'):
        '''  data: {'loss': value,  'dice': value}  '''

        loss, dice = data['loss'], data['dice']
        if iter < 0: # epoch losses
            self.writer.add_scalar('epoch/%s_loss' % (phase), loss, epoch)
            self.writer.add_scalar('epoch/%s_dice' % (phase), dice, epoch)
            if 'dice_bg' in data.keys():
                self.writer.add_scalar('epoch/%s_dice_bg' % (phase), data['dice_bg'], epoch)

        else:   # iter losses
            self.writer.add_scalar('iter/%s_loss' % (phase), loss, iter)
            self.writer.add_scalar('iter/%s_dice' % (phase), dice, iter)
            if 'dice_bg' in data.keys():
                self.writer.add_scalar('iter/%s_dice_bg' % (phase), data['dice_bg'], iter)

    def plot_curves_multi(self, data, epoch, iter=-1, phase='train'):
        ''' data: multiple loss terms '''
        ignore_keys = ['pce_verbose', 'size_verbose']
        group_name = 'epoch_verbose' if iter < 0 else 'iter_verbose'
        count = epoch if iter < 0 else iter

        for key, value in data.items():
            if key in ignore_keys:
                continue
            self.writer.add_scalar('%s/%s_%s' % (group_name, phase, key), value, count)

    def plot_test_curves(self, data, iter=1, phase='test', avg=False):
        '''  data: {'dice': value, 'dice_bg': value}  '''
        dice = data['dice']     # dice_bg = data['dice_bg']
        if avg is False:
            self.writer.add_scalar('%s/dice' % (phase), dice, iter)
            # self.writer.add_scalar('%s/dice_bg' % (phase), dice_bg, iter)
        else:
            self.writer.add_scalar('%s/avg_dice' % (phase), dice, iter)
            # self.writer.add_scalar('%s/avg_dice_bg' % (phase), dice_bg, iter)


    def log_init0(self):
        logging.basicConfig(filename=opt.log_dir + "/log_{}.txt".format(args.id), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(opt))
        logging.info('Time: %s' % datetime.datetime.now())

    def log_init(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        logfile = opt.log_dir + "/log_{}.txt".format(args.id)
        fh = logging.FileHandler(logfile)#, mode='w') # whether to clean previous file
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        logging.info(str(opt))
        logging.info('Time: %s' % datetime.datetime.now())
        # print(opt)

    def log_batch(self, data, epoch, i_batch):
        loss, dice = data['loss'], data['dice']
        phrase = 'Epoch: {:4.0f} i_batch: {:4.0f} mDice: {:.6f} Loss: {:.6f} '.format(
            epoch, i_batch, dice, loss)
        if args.demo == '':
            logging.info(phrase)
        else:
            logging.warning(phrase)
    def log_batch_two_loss(self, data, epoch, i_batch, loss_terms=['weak_loss', 'pseudo_loss']):
        loss, dice = data['loss'], data['dice']
        phrase = 'Epoch: {:4.0f} i_batch: {:4.0f} mDice: {:.6f} Loss: {:.6f} {}: {:.6f} {}: {:.6f}'.format(
            epoch, i_batch, dice, loss, loss_terms[0], data[loss_terms[0]], loss_terms[1], data[loss_terms[1]])
        if args.demo == '':
            logging.info(phrase)
        else:
            logging.warning(phrase)

    def log_epoch(self, data, epoch, phase, extra_key=None):
        loss, dice = data['loss'], data['dice']
        logging.warning(
            '{} Epoch: {:4.0f} mDice: {:.6f} Loss: {:.6f}'.format(
            phase, epoch, dice, loss))
        if phase == 'val':
            logging.warning(
                'Best_mDice: {:.6f} Lowest_val_loss: {:.6f} AtEpoch: {:f}'.format(self.best['dice'], self.best['loss'], self.best['epoch']))
            if extra_key is not None:
                logging.warning('Best {}: {:.6f}'.format(extra_key, self.best[extra_key]))
            time_elapsed = time.time() - self.since
            logging.warning('Time till now {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
            logging.warning('Time: %s' % datetime.datetime.now())

    def log_final(self):
        time_elapsed = time.time() - self.since
        logging.warning('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
    
    def update_pseudo_by_feature(self, pseudos, features):
        if opt.model.prototype is not None:
            pseudos = self.update_pseudo_by_feature_sparse_global(pseudos, features)
        else:
            pseudos = self.update_pseudo_by_feature_dense_local(pseudos, features)
        return pseudos

    def update_pseudo_by_feature_sparse_global(self, pseudos, features):
        # ipdb.set_trace()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        shape = pseudos.shape
        pseudos = pseudos.reshape(-1)
        unlabel_ids = torch.where(pseudos == 255)[0]
        inds_fgs, inds_bgs = [], []
        for feat in features:
            ratio = opt.model.recover_ratio
            if len(inds_fgs) == 0:
                ratio = ratio / 2
            inds = self.recover_inds(feat, pseudos, cos, ratio)
            inds_fgs.append(inds[0])
            inds_bgs.append(inds[1])
        inds_shape = unlabel_ids.shape
        inds_fg = self.intersection_of_inds(inds_fgs, inds_shape)
        inds_bg = self.intersection_of_inds(inds_bgs, inds_shape)
        pseudos[unlabel_ids[inds_fg]] = 1
        pseudos[unlabel_ids[inds_bg]] = 0
        # print(inds_fg.shape[0], inds_bg.shape[0], inds_fg.shape[0] + inds_bg.shape[0], inds_shape)
        # re_fg = (sim_fg - sim_bg) > opt.model.recover_ratio
        # re_bg = (sim_bg - sim_fg) > opt.model.recover_ratio
        # pseudos[unlabel_ids[re_fg]] = 1
        # pseudos[unlabel_ids[re_bg]] = 0
        pseudos = pseudos.reshape(shape)
        return pseudos

    def intersection_of_inds(self, inds_n, inds_shape):
        inds_bool = torch.ones(inds_shape).bool().cuda()
        for inds_i in inds_n:
            bool_inds_i = torch.zeros(inds_shape).bool().cuda()
            bool_inds_i[inds_i] = True
            inds_bool = inds_bool * bool_inds_i
        return inds_bool
    
    def recover_inds(self, features, pseudos, cos, ratio):
        features = features.reshape(features.shape[-4], -1).transpose(0, 1)
        unlabel_ids = torch.where(pseudos == 255)[0]
        unlabel_feat = features[unlabel_ids]
        fg_feat = features[torch.where(pseudos == 1)]
        bg_feat = features[torch.where(pseudos == 0)]
        fg_v = fg_feat.sum(dim=0, keepdim=True) / fg_feat.shape[0]
        bg_v = bg_feat.sum(dim=0, keepdim=True) / bg_feat.shape[0]
        sim_fg = cos(fg_v, unlabel_feat)
        sim_bg = cos(bg_v, unlabel_feat)
        _, inds_fg = torch.topk(sim_fg - sim_bg, int(ratio * unlabel_feat.shape[0]))
        _, inds_bg = torch.topk(sim_bg - sim_fg, int(ratio * unlabel_feat.shape[0]))
        # _, inds_fg = torch.topk(sim_fg - sim_bg, int(opt.model.recover_ratio * unlabel_feat.shape[0]))
        # _, inds_bg = torch.topk(sim_bg - sim_fg, int(opt.model.recover_ratio * unlabel_feat.shape[0]))
        return inds_fg, inds_bg

    def update_pseudo_by_feature_dense_local(self, pseudos, features):
        # ipdb.set_trace()
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for ii in range(opt.model.feat_round):
            pseudos_update = copy.deepcopy(pseudos)
            xids, yids, zids = torch.where(pseudos[0] == 255)
            for i in range(len(xids)):
                x, y, z = xids[i], yids[i], zids[i]
                si = opt.model.feat_kernel_size // 2
                ei = opt.model.feat_kernel_size - si
                feat = copy.deepcopy(features[0][0, :, x:x + 1, y:y + 1, z:z + 1])
                feat_local = copy.deepcopy(features[0][0, :, x - si:x + ei, y - si:y + ei, z - si:z + ei])
                pseudos_local = copy.deepcopy(pseudos[0, x - si:x + ei, y - si:y + ei, z - si:z + ei]).flatten()
                sim = cos(feat, feat_local).flatten()   # feat.repeat(1,3,3,3)
                if len(sim) < opt.model.feat_topk:
                    # print(x, y, z, pseudos.shape, sim, opt.model.feat_topk)
                    continue
                _, inds = torch.topk(sim, opt.model.feat_topk)
                pseudos_topk = pseudos_local[inds]
                nbg = (pseudos_topk == 0).int().sum()
                nfg = (pseudos_topk == 1).int().sum()
                if nbg > nfg * 2:
                    pseudos_update[0, x, y, z] = 0
                if nfg > nbg * 2:
                    pseudos_update[0, x, y, z] = 1
            pseudos = pseudos_update
        return pseudos

    def E_model_generate_pseudo(self, fp_filter_ratio, fn_filter_ratio, filter_k_times):
        # inference train set, save to a fix dir, for later dataloader.
        # 0. Params for uncertainty filter
        save_em_pseudo_dir = opt.data.em_save_pseudo_dir + '/%s/' % args.id
        if not os.path.exists(save_em_pseudo_dir):
            os.makedirs(save_em_pseudo_dir)
        # 1. dataloader, model forward
        for i_batch, data in enumerate(self.E_train_loader):
            inputs = data['image'].float().cuda()
            with torch.no_grad():
                outputs = self.model(inputs, opt.model.feat_for_pseudo)
                if opt.model.feat_for_pseudo is not None:
                    features, outputs = outputs
                probs = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]  # preds (B, H, W)

            # 1.1 ae_pred, uncertainty filter,
            weak_label = data['weak_label'][0].long().cuda()
            filter_phase = opt.model.filter_phase  # {'prob', 'ae', 'sdf', 'sdfbg', 'probae', 'probsdf'}
            filter_extra_mode = opt.model.filter_extra
            # print('1Time: %s' % datetime.datetime.now())
            pseudos = self.filter2pseudo(probs, logits=outputs[0], filter_phase=filter_phase, weak_labels=weak_label, extra_mode=filter_extra_mode,
                      fp_filter_ratio=fp_filter_ratio, fn_filter_ratio=fn_filter_ratio, filter_k_times=filter_k_times)
            # semi-3d pseudo label refinement
            # print('2Time: %s' % datetime.datetime.now())
            if opt.model.feat_for_pseudo is not None:
                pseudos = self.update_pseudo_by_feature(pseudos, features)
            # print('3Time: %s' % datetime.datetime.now())
            # 1.2 save mat, be careful of filename
            case_id = self.save_val_ids['train'][i_batch]
            save_path = save_em_pseudo_dir + case_id + '.mat'
            pseudo = pseudos[0].cpu().numpy().astype(np.uint8)
            # modify initial weak label
            if filter_extra_mode == 'modify_initial_weak':
                weak_np = weak_label[0, 0].cpu().numpy().astype(np.uint8)
                weak_np[np.logical_and(weak_np == 1, pseudo != 1)] = 255
            if not args.save_em_init:
                if filter_extra_mode == 'modify_initial_weak':
                    savemat(save_path, {'pred': pseudo, 'weak_label': weak_np})
                else:
                    savemat(save_path, {'pred': pseudo})
            else:
                vis_save_dir = save_em_pseudo_dir + 'vis/'
                if not os.path.exists(vis_save_dir):
                    os.makedirs(vis_save_dir)
                save_path = vis_save_dir + case_id + '.nii.gz'
                nib.save(nib.Nifti1Image(pseudo, np.eye(4)), save_path)
                if filter_extra_mode == 'modify_initial_weak':
                    nib.save(nib.Nifti1Image(weak_np, np.eye(4)), vis_save_dir + case_id + '_weak.nii.gz')

        logging.warning('E step done: generated new pseudo labels for M step.')
        if args.save_em_init:   # breakdown because only generating init pseudo label
            sys.exit()


    def vis(self):
        if 'val' in self.phases:
            print('Num per valid_loader', sum(1 for _ in self.data_loaders['val']))
        if 'test' in self.phases:
            print('Num per valid_loader', sum(1 for _ in self.data_loaders['test']))
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        dice_meter_ae = MultiLossMeter()    # used only when ae_model employed
        self.model.train(False)

        for phase in self.phases:
            save_dir = os.path.join(os.getcwd(), '../output/%s_%s' % (args.id, phase))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            log_path = os.path.join(save_dir, 'log.txt')
            log = open(log_path, 'w')
            log.write('Dice metrics\n')
            if self.ae_model is not None:
                log_ae_path = os.path.join(save_dir, 'log_ae.txt')
                log_ae = open(log_ae_path, 'w')
                log_ae.write('Dice metrics\n')

            loss_meter.reset(); dice_meter.reset(); dice_meter_ae.reset()
            probs = []
            dice_list = []; ae_dice_list = []    # for calculating std

            if not opt.data.use_conf:       # default choice
                for i_batch, data in tqdm(enumerate(self.data_loaders[phase])):
                    if phase != 'test':
                        gt = [it.long().cuda() for it in data['gt']]
                    if opt.train.loss_name == 'MultipleOutputLoss2':
                        if opt.ae.input is not None:
                            targets = data['gt']
                        else:
                            targets = data['weak_label'] if opt.data.use_weak else data['gt']
                        targets = [it.long().cuda() for it in targets]  # deep supervision
                        if opt.ae.input == 'gt':
                            inputs = self.get_input_for_auto_encoder(targets)
                        elif opt.ae.input == 'pred_aug':
                            inputs = data['pred'][0].float().cuda()
                        else:
                            inputs = data['image'].float().cuda()
                        with torch.no_grad():
                            # before = datetime.datetime.now()
                            outputs = self.model(inputs)
                            # print('cost time:', datetime.datetime.now() - before)
                            preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]
                            # TODO: AE post_process
                            if self.ae_model is not None:
                                ae_inputs = torch.unsqueeze((preds > 0.5).float(), dim=1)
                                ae_outputs = self.ae_model(ae_inputs)
                                ae_preds = nn.Softmax(dim=1)(ae_outputs[0])[:, 1, ...]
                                if phase != 'test':
                                    dices_ae, dice_names_ae = self.metric.forward(ae_preds.cpu(), gt[0].cpu())
                                    dice_meter_ae.update(dices_ae, dice_names_ae)
                                else:
                                    dices_ae = [torch.tensor(0.00, requires_grad=True)]
                                    dice_meter_ae.update(dices_ae, ['dice'])
                                ae_dice_list.append(dices_ae[0])

                            # # forward twice
                            # preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]
                            # preds = (preds > 0.5).float()
                            # preds = torch.unsqueeze(preds, 1)
                            # outputs = self.model(preds)

                        # #TODO: temporary debug code
                        # if opt.model.pseudo_wce != '':
                        #     em_wce_phase = opt.model.pseudo_wce
                        #     logits = nn.Softmax(dim=1)(outputs[0])
                        #     pseudos, pseudos_weights = self.filter2pseudo_weights(logits, wce_phase=em_wce_phase, weak_labels=None)

                        if phase != 'test':
                            dices, dice_names = self.metric.forward(preds.cpu(), gt[0].cpu())
                        else:
                            dices = [torch.tensor(0.00, requires_grad=True)]; dice_names = ['dice']
                        dice_list.append(dices[0])
                        if phase != 'test':
                            gt = gt[0][:, 0, ...]
                    else:
                        preds = nn.Softmax(dim=1)(outputs)[:, 1, ...]
                        dices, dice_names = self.metric.forward(preds.cpu(), gt.cpu())
                    dice_meter.update(dices, dice_names)

                    # self.plot_test_curves({'dice': dices[0]}, iter=i_batch)     #'dice_bg': dices[1]
                    _id = self.val_ids[phase][i_batch] if self.val_ids is not None else i_batch
                    # save volume
                    if phase != 'test':
                        self.save_pred([inputs.cpu(), (preds>0.5).detach().cpu(), gt.int().cpu()], names=['image', 'pred', 'gt_label'],
                                       id=_id, title=args.id, phase=phase, save_volume='3d')
                    else:
                        self.save_pred([inputs.cpu(), (preds > 0.5).detach().cpu()], names=['image', 'pred'],
                                       id=_id, title=args.id, phase=phase, save_volume='3d')
                    if self.ae_model is not None:   # save ae processed result
                        self.save_pred([(preds>0.5).cpu(), (ae_preds > 0.5).detach().cpu()], names=['ae_input', 'ae_pred'],
                                   id=_id, title=args.id, phase=phase, save_volume='3d')
                        log_ae.write('%s: dice_fg %.4f\n' % (_id, dices_ae[0]))
                    # save heatmap
                    if args.save_heatmap:
                        pass
                        self.save_pred([preds.detach().cpu()], names=['heatmap'],
                                       id=_id, title=args.id, phase=phase, save_volume='3d')
                    # # TODO: temporary debug code
                    # if opt.model.pseudo_wce != '':
                    #     self.save_pred([pseudos_weights.cpu()], names=['pweight'], id=_id, title=args.id, phase=phase, save_volume='3d')

                    if not opt.data.use_entropy:
                        # probability-based uncertainty
                        threshold = 0.5 # 0.2 modified in 1028
                        avg_prob = (preds[preds >= threshold]).sum() / (preds >= threshold).sum()
                        probs.append(avg_prob)
                    '''
                    else :
                        # entropy-based uncertainty
                        T = 8
                        inputs_r = inputs.repeat(2, 1, 1, 1, 1)
                        stride = inputs.shape[0]
                        _d, _h, _w = inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
                        preds_eu = torch.zeros([stride * T, 2, _d, _h, _w]).cuda()
                        for i in range(T // 2):
                            ema_inputs = inputs_r + torch.randn_like(inputs_r) * 0.1
                            with torch.no_grad():
                                out = self.model(ema_inputs)
                                preds_eu[2 * stride * i:2 * stride * (i + 1)] = out[0]
                        preds_eu = nn.functional.softmax(preds_eu, dim=1)
                        preds_eu = preds_eu.reshape(T, stride, 2, _d, _h, _w)
                        preds_eu = torch.mean(preds_eu, dim=0)  # (batch, 2, 112,112,80)
                        uncertainty = -1.0 * torch.sum(preds_eu * torch.log(preds_eu + 1e-6), dim=1, keepdim=True)  # (batch, 1, 112,112,80)
                        confidence = 1 - uncertainty
                        self.save_pred([confidence.cpu()[:, 0, ...]], names=['entropy_conf'], id=_id, title=args.id, phase=phase, save_volume='3d')
                        threshold = 0.8
                        # avg_prob = (confidence[confidence >= threshold]).sum() / (confidence >= threshold).sum()
                        avg_prob = (confidence[confidence < threshold]).sum() / (preds>0.5).sum()
                        print(avg_prob)
                        probs.append(avg_prob)

                        # import nibabel as nib
                        # path = '/group/lishl/weak_exp/output/0731_trachea_gen_entropy_val/entropy_conf/Case25_00.nii.gz'
                        # nib.save(nib.Nifti1Image(confidence.cpu().numpy()[0, 0], np.eye(4)), path)

                        # all_prob = (confidence[confidence >= 0.001]).sum() / (confidence >= 0.001).sum()
                        # print('0.001_avg: ', all_prob, '0.2_avg', avg_prob)
                    '''
                    log.write('%s: dice_fg %.4f\t avg_prob: %.5f\n' % (_id, dices[0], avg_prob))  #dice_bg: %.4f\t dices[1],

            '''
            else:
                for _iter in range(opt.data.conf_iter):
                    for i_batch, data in tqdm(enumerate(self.data_loaders[phase])):
                        inputs = data['image'].float().cuda()
                        with torch.no_grad():
                            outputs = self.model(inputs)  # need modifying when model changes

                        # extract real gt for weak setting
                        gt = [it.long().cuda() for it in data['gt']]
                        if opt.train.loss_name == 'MultipleOutputLoss2':
                            preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]
                            dices, dice_names = self.metric.forward(preds.cpu(), gt[0].cpu())
                            gt = gt[0][:, 0, ...]
                        else:
                            preds = nn.Softmax(dim=1)(outputs)[:, 1, ...]
                            dices, dice_names = self.metric.forward(preds.cpu(), gt.cpu())
                        dice_meter.update(dices, dice_names)

                        self.plot_test_curves({'dice': dices[0]}, iter=i_batch)     #, 'dice_bg': dices[1]
                        _id = self.val_ids[phase][i_batch] if self.val_ids is not None else i_batch
                        _id = _id + '_%s' % str(_iter)
                        # save volume
                        self.save_pred([inputs.cpu(), (preds > 0.5).detach().cpu(), gt.int().cpu()],
                                       names=['image', 'pred', 'gt_label'],
                                       id=_id, title=args.id, phase=phase, save_volume='3d')
                        # # save heatmap
                        # self.save_pred([preds.detach().cpu(), gt.int().cpu()], names=['pred', 'gt_label'],
                        #                id=_id, title=args.id, phase=phase, save_heatmap=True)
                        threshold = 0.2
                        avg_prob = (preds[preds >= 0.2]).sum() / (preds >= 0.2).sum()
                        probs.append(avg_prob)
                        log.write('%s: dice_fg %.4f\t dice_bg: %.4f\t avg_prob: %.3f\n' % (_id, dices[0], dices[1], avg_prob))
            '''
            avg_terms = loss_meter.get_metric()
            dice_terms = dice_meter.get_metric()
            avg_terms = {**avg_terms, **dice_terms}
            # self.plot_test_curves({'dice': avg_terms['dice']}, avg=True)    #, 'dice_bg': avg_terms['dice_bg']
            log.write('\nAverage dices\n')
            log.write('AVG: dice_fg %.4f\t  median_prob: %.3f\n' % (avg_terms['dice'], sorted(probs)[len(probs)//2]))  #dice_bg: %.4f\t  avg_terms['dice_bg'],
            log.write('Samples std: %.4f\n' % torch.std(torch.tensor(dice_list)).item())
            log.close()
            if self.ae_model is not None:
                dice_ae_terms = dice_meter_ae.get_metric()
                log_ae.write('\nAverage dices\n')
                log_ae.write('AVG: dice_fg %.4f\n' % (dice_ae_terms['dice']))
                log_ae.write('Samples std: %.4f\n' % torch.std(torch.tensor(ae_dice_list)).item())
                log_ae.close()

            # print metrics
            print('METRICS:')
            for key, value in avg_terms.items():
                print(key, value)
            print()

    def mask_sp_target(self, sp_targets, outputs):
        mask_ratios = ast.literal_eval(opt.sp.mask_ratios)
        ratio_low, ratio_up = mask_ratios
        with torch.no_grad():
            if opt.train.loss_name == 'MultipleOutputLoss2':
                for d in range(len(outputs)):
                    # label_nib = nib.Nifti1Image(copy.deepcopy(sp_targets[d][0,0]).cpu().numpy().astype(np.uint8), np.eye(4))
                    # nib.save(label_nib, os.path.join('.', 'sp{}.nii.gz'.format(d)))
                    # mask = outputs[d]
                    mask = (nn.Softmax(dim=1)(outputs[d])[:, 1, ...] > 0.5).long().unsqueeze(dim=1)
                    mask_shape = mask.shape

                    mask_low = ndimage.zoom(mask.cpu().clone(), zoom=[1.0, 1.0, ratio_low, ratio_low, ratio_low], order=0)
                    mask_low_shape = mask_low.shape
                    shape_delta_low = [mask_shape[-i] - mask_low_shape[-i] for i in range(1,4)]
                    pad_size = [shape_delta_low[0] // 2, shape_delta_low[0] - shape_delta_low[0] // 2,
                                shape_delta_low[1] // 2, shape_delta_low[1] - shape_delta_low[1] // 2,
                                shape_delta_low[2] // 2, shape_delta_low[2] - shape_delta_low[2] // 2]
                    pad_low = nn.ConstantPad3d(pad_size, 0)   #inverse
                    mask_low = pad_low(torch.Tensor(mask_low))

                    mask_up = ndimage.zoom(mask.cpu().clone(), zoom=[1.0, 1.0, ratio_up, ratio_up, ratio_up], order=0)
                    mask_up_shape = mask_up.shape
                    shape_delta_up = [mask_up_shape[-i] - mask_shape[-i] for i in reversed(range(1,4))]
                    crop_idx = [shape_delta_up[0] // 2, mask_shape[-3] + shape_delta_up[0] // 2,
                                shape_delta_up[1] // 2, mask_shape[-2] + shape_delta_up[1] // 2,
                                shape_delta_up[2] // 2, mask_shape[-1] + shape_delta_up[2] // 2]
                    mask_up = mask_up[:, :, crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3], crop_idx[4]:crop_idx[5]]
                    mask_up = torch.Tensor(mask_up)
                    
                    mask_merge = (mask_up - mask_up * mask_low).cuda().long()

                    sp_target = sp_targets[d]
                    sp_target = mask_merge * sp_target + (mask_merge == 0).long() * 255
                    sp_targets[d] = sp_target
                    
                    # label_nib = nib.Nifti1Image(copy.deepcopy(mask_merge[0,0]).cpu().numpy().astype(np.uint8), np.eye(4))
                    # nib.save(label_nib, os.path.join('.', 'mask{}.nii.gz'.format(d)))
                    # label_nib = nib.Nifti1Image(copy.deepcopy(sp_target[0,0]).cpu().numpy().astype(np.uint8), np.eye(4))
                    # nib.save(label_nib, os.path.join('.', '{}.nii.gz'.format(d)))
            else:
                mask = (nn.Softmax(dim=1)(outputs)[:, 1, ...] > 0.5).long().unsqueeze(dim=1)
                mask_shape = mask.shape

                mask_low = ndimage.zoom(mask.cpu().clone(), zoom=[1.0, 1.0, ratio_low, ratio_low, ratio_low], order=0)
                mask_low_shape = mask_low.shape
                shape_delta_low = [mask_shape[-i] - mask_low_shape[-i] for i in range(1,4)]
                pad_size = [shape_delta_low[0] // 2, shape_delta_low[0] - shape_delta_low[0] // 2,
                            shape_delta_low[1] // 2, shape_delta_low[1] - shape_delta_low[1] // 2,
                            shape_delta_low[2] // 2, shape_delta_low[2] - shape_delta_low[2] // 2]
                pad_low = nn.ConstantPad3d(pad_size, 0)   #inverse
                mask_low = pad_low(torch.Tensor(mask_low))

                mask_up = ndimage.zoom(mask.cpu().clone(), zoom=[1.0, 1.0, ratio_up, ratio_up, ratio_up], order=0)
                mask_up_shape = mask_up.shape
                shape_delta_up = [mask_up_shape[-i] - mask_shape[-i] for i in reversed(range(1,4))]
                crop_idx = [shape_delta_up[0] // 2, mask_shape[-3] + shape_delta_up[0] // 2,
                            shape_delta_up[1] // 2, mask_shape[-2] + shape_delta_up[1] // 2,
                            shape_delta_up[2] // 2, mask_shape[-1] + shape_delta_up[2] // 2]
                mask_up = mask_up[:, :, crop_idx[0]:crop_idx[1], crop_idx[2]:crop_idx[3], crop_idx[4]:crop_idx[5]]
                mask_up = torch.Tensor(mask_up)
                    
                mask_merge = (mask_up - mask_up * mask_low).cuda().long()

                sp_targets = mask_merge * sp_targets + (mask_merge == 0).long() * 255
        return sp_targets

    def train_superpixel(self):
        num_epochs = opt.train.n_epochs
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        total_iter = {'train': 0, 'val': 0, 'test': 0}
        # warmup dataloader

        # print('Num per train_loader', sum(1 for _ in self.data_loaders['train']))
        # print('Num per valid_loader', sum(1 for _ in self.data_loaders['val']))

        # val_sample = next(self.data_loaders['val'])
        # del val_sample

        for epoch in tqdm(range(num_epochs)):
            for phase in self.phases:
                if phase == 'train' or phase == 'train_val':
                    self.model.train(True)
                else:
                    self.model.train(False)
                    sum(1 for _ in self.data_loaders['val'])    # for debugging val_loader error: unexpected number
                loss_meter.reset(); dice_meter.reset()
                count_batch = 0

                while count_batch < opt.train.n_batches:
                    for i_batch, data in enumerate(self.data_loaders[phase]):
                        # import ipdb; ipdb.set_trace()   # For AEDataset, data.keys(): ['pred', 'image', 'gt']

                        
                        if opt.train.loss_name == 'MultipleOutputLoss2':
                            # target supervision
                            if opt.ae.input is not None:
                                targets = data['gt']
                            elif opt.model.use_model1:
                                # generated preds
                                inputs = data['image'].float().cuda()
                                with torch.no_grad():
                                    if opt.model.model1_fchannel > 0:
                                        Features, F_outputs = self.model1(inputs)   # Features: (B,64, ...)
                                    else:
                                        F_outputs = self.model1(inputs)
                                    F_preds = [nn.Softmax(dim=1)(F_output)[:, 1, ...] for F_output in F_outputs]
                                    targets = F_preds
                            else:
                                targets = data['weak_label'] if opt.data.use_weak else data['gt']
                            targets = [it.long().cuda() for it in targets]  # deep supervision
                            if opt.sp.use_type != '':
                                sp_targets = data['sp_edge']
                                sp_targets = [it.long().cuda() for it in sp_targets]  # deep supervision
                            # input
                            if opt.ae.input == 'gt':
                                inputs = self.get_input_for_auto_encoder(targets)
                            elif opt.ae.input == 'pred_aug':
                                inputs = data['pred'][0].float().cuda()
                            elif opt.model.use_model1:
                                if opt.model.model1_fchannel > 0:
                                    inputs = Features
                                else:
                                    inputs = data['image'].float().cuda()
                            else:
                                inputs = data['image'].float().cuda()
                            outputs = self.model(inputs)
                            if opt.sp.use_type != '':
                                sp_outputs = outputs[1]
                                outputs = outputs[0]
                                if 'regress' in opt.sp.use_type:
                                    sp_preds = sp_outputs[0]
                                else:
                                    sp_preds = nn.Softmax(dim=1)(sp_outputs[0])[:, 1, ...]  # preds (B, H, W)
                            preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]  # preds (B, H, W)
                        else:
                            inputs = data['image'].float().cuda()
                            targets = data['weak'] if opt.data.use_weak else data['gt']
                            targets = targets.long().cuda()
                            outputs = self.model(inputs)
                            if opt.sp.use_type != '':
                                sp_outputs = outputs[1]
                                outputs = outputs[0]
                                if 'regress' in opt.sp.use_type:
                                    sp_preds = sp_outputs
                                else:
                                    sp_preds = nn.Softmax(dim=1)(sp_outputs)[:, 1, ...]  # preds (B, H, W)
                            preds = nn.Softmax(dim=1)(outputs)[:, 1, ...]  # preds (B, H, W)

                        if opt.sp.mask_ratios != '':
                            # targets = data['gt']
                            # targets = [it.long().cuda() for it in targets]  # deep supervision
                            # sp_targets = self.mask_sp_target(sp_targets, targets)
                            sp_targets = self.mask_sp_target(sp_targets, outputs)

                        # train mode, val mode
                        if (phase == 'train' or phase == 'train_val'):
                            losses, loss_names = self.criterion(outputs, targets)
                            loss = losses[0]
                            if opt.sp.use_type != '':
                                sp_losses, sp_loss_names = self.sp_criterion(sp_outputs, sp_targets)
                                sp_loss = sp_losses[0]
                                loss = loss + sp_loss * opt.sp.loss_weight
                                losses += sp_losses
                                loss_names += ['sp_' + sp_loss_name for sp_loss_name in sp_loss_names]
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        else:
                            with torch.no_grad():
                                losses, loss_names = self.criterion(outputs, targets)
                                if opt.sp.use_type != '':
                                    sp_losses, sp_loss_names = self.sp_criterion(sp_outputs, sp_targets)
                            loss = losses[0]
                            if opt.sp.use_type != '':
                                sp_loss = sp_losses[0]
                                loss = loss + sp_loss * opt.sp.loss_weight
                                losses += sp_losses
                                loss_names += ['sp_' + sp_loss_name for sp_loss_name in sp_loss_names]
                        loss_meter.update(losses, loss_names)

                        # ipdb.set_trace()
                        # print(inputs[0, 0, 0, 0, 0], outputs[0][0, 0, 0, 0, 0], loss)
                        # print(inputs[1, 0, 0, 0, 1], outputs[0][1, 0, 0, 0, 1], loss)

                        # extract real gt for weak setting
                        gt = data['gt']
                        if opt.train.loss_name == 'MultipleOutputLoss2':
                            dices, dice_names = self.metric.forward(preds.cpu(), gt[0].cpu())
                            if opt.sp.use_type != '' and 'regress' not in opt.sp.use_type:
                                sp_dices, sp_dice_names = self.metric.forward(sp_preds.cpu(), sp_targets[0].cpu())
                                dices += sp_dices
                                dice_names += ['sp_' + sp_dice_name for sp_dice_name in sp_dice_names]
                        else:
                            dices, dice_names = self.metric.forward(preds.cpu(), gt.cpu())
                            if opt.sp.use_type != '' and 'regress' not in opt.sp.use_type:
                                sp_dices, sp_dice_names = self.metric.forward(sp_preds.cpu(), sp_targets.cpu())
                                dices += sp_dices
                                dice_names += ['sp_' + sp_dice_name for sp_dice_name in sp_dice_names]
                        dice_meter.update(dices, dice_names)

                        # plot running loss curve
                        metrics = {'loss': loss, 'dice': dices[0], 'dice_bg': dices[1]}
                        # self.plot_curves(metrics, None, total_iter[phase], phase=phase)
                        total_iter[phase] += 1
                        self.log_batch(metrics, epoch, i_batch)

                        # plot multiple loss curve
                        # loss_terms = dict(zip(loss_names, losses))
                        # self.plot_curves_multi(loss_terms, None, total_iter[phase], phase=phase)

                        count_batch += 1
                        if count_batch >= opt.train.n_batches:
                            break

                        # release memory
                        del inputs, targets, gt, preds, outputs, losses, dices
                        if opt.sp.use_type != '':
                            del sp_targets, sp_outputs

                    if phase == 'val':
                        break

                avg_terms = loss_meter.get_metric()
                dice_terms = dice_meter.get_metric()
                avg_terms = {**avg_terms, **dice_terms}
                self.plot_curves_multi(avg_terms, epoch, phase=phase)

                if phase == 'val' or phase == 'train_val':
                    if avg_terms['dice'] > self.best['dice']:
                        self.update_best(avg_terms, epoch)
                self.log_epoch(avg_terms, epoch, phase)

            # # release memory
            # del inputs, targets, gt, preds, outputs, losses, dices

            if epoch % opt.snapshot == 0:
                snapshot = copy.deepcopy(self.model)
                torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.model_dir, epoch))

            # adjust lr on epoch level
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_terms['dice'])
                else:
                    self.scheduler.step()
            elif opt.train.lr_decay == 'poly':
                lr_ = opt.train.lr * (1 - epoch / opt.train.n_epochs) ** 0.9  # power=0.9
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

        self.log_final()
        self.writer.close()





import yaml
from easydict import EasyDict as edict


config = edict()


# 1. data_dir
config.snapshot = 10

config.data_dir = ''
config.model_dir = ''
config.log_dir = ''
config.tb_dir = ''


# 2. data related
config.data = edict()
config.data.data_dir = ''
config.data.dataname = ''
config.data.use_weak = False     # fully/weak supervision
config.data.scribble = False     # weak label format: box/scribble

config.data.num_workers = 4
config.data.use_conf = False    # select part of confident samples
config.data.use_entropy = False # use entropy as uncertainty measure

config.data.use_select = False
config.data.select_label_percent = 30
config.data.select_use_pred_num = 5
config.data.weak_label_dir = ''
config.data.em_save_pseudo_dir = ''

config.data.write_log_batch = True      # logging per batch

# 3. model related
config.model = edict()
config.model.network = 'VNet'
config.model.pre_model = ''
config.model.use_author_block = False
## Generic UNet parameter
config.model.pool_op_kernel_sizes = ''
config.model.conv_kernel_sizes = ''
config.model.disable_skip = False
config.model.deep_supervision = True

config.model.use_model1 = False
config.model.model1_path = ''
config.model.model1_fchannel = 0
config.model.use_model2 = False
config.model.use_finetune = False
config.model.finetune_model_path = ''
config.model.finetune_model_path_ae = ''

config.model.use_emiter = False     # em forward in some iteration or epoch
config.model.iterate_epoch = 0  # em forward in some epoch, 0 for not using this
config.model.fpr_drop_freq = 0  # curriculum setting: the frequency of fp_filter_ratio dropping from 0.9->0.4, (epochs)
config.model.fpr_drop_rate = 0.1 # default fpr drop rate per drop_freq
config.model.fpr_min_ratio = 0.1 # dropping lower bound
config.model.pseudo_wce = ''  # loss: pixel-wise weight calculation {'ae_wce'}
config.model.pseudo_wce_func = ''   # weight function {'sigmoid', 'exponent'}
config.model.pseudo_wce_powk = 2    # power k: prob**k
config.model.pseudo_wce_mu   = 0    # mu: mean value of sigmoid function, only used for 'sigmoid' method
config.model.filter_phase = ''  # EM filter policy: {'t_scaling', 'prob', 'ae', 'ae_psrank', 'sdf', 'sdfbg', 'probae', 'probae_addrank', 'probsdf'}
config.model.filter_extra = ''  # extra mode, {'trust_aebg'}

config.model.fp_filter_ratio = 0
config.model.fn_filter_ratio = 0
config.model.filter_k_times = 0
config.model.psrk = 1.0     # only for 'ae_psrank', combined weight of prob and sdf ranking
config.model.feat_for_pseudo = None
config.model.feat_kernel_size = 0
config.model.feat_topk = 0
config.model.feat_round = 0
config.model.prototype = None
config.model.recover_ratio = 0
config.model.t_scaling_t_prob = 1
config.model.t_scaling_t_ae = 1
config.model.t_scaling_combine = 'add'


# 4. training params
config.train = edict()
config.train.n_epochs = 1000
config.train.n_batches = 250
config.train.train_batch = 2
config.train.valid_batch = 2
config.train.test_batch = 1

config.train.pseudo_one_supervision = False  # one supervision but not deep supervision for pseudo masks
config.train.only_pseudo_loss = False   # only pseudo loss in EM iterations
config.train.log_two_loss = False

config.train.loss_name = ''
config.train.focal_loss = False
config.train.ind_pseudo_loss = ''     # independent pseudo loss: default the same as other loss. can set {'ce', 'focal'}
config.train.with_dice_loss = 0
config.train.graphcut = False
config.train.fg_weight = 1
config.train.auto_weight_fg_bg = False
config.train.pseudo_auto_weight = True
config.train.max_fg_weight = 1000000     # max fg weight: to void too sensitive training
config.train.draw_fgbg = False
config.train.alpha = 0
config.train.beta = 0
config.train.gamma = 0
config.train.min_lr = 0
config.train.pseudo_loss_weight = 1
config.train.weak_loss_weight = 1

config.train.lr = 1e-2
config.train.lr_decay = 'constant'
config.train.momentum = 0.9
config.train.weight_decay = 1e-4

config.train.milestone = ''
config.train.gamma = 0.1
config.train.plateau_patience = 3
config.train.plateau_gamma = 0.1


# 5. similarity based soft_label
config.similarity = edict()
config.similarity.use = False
config.similarity.lambda_1 = 1
config.similarity.top_thresh = 0.1
config.similarity.auto_weight = True
config.similarity.init_epoch = 0
config.similarity.visualize = False
config.similarity.component = 'simi_dist'
config.similarity.label_channel = 'dependent'
config.similarity.e_type = ''


# 6. label sparsity
config.label = edict()
config.label.divided_by = 1



# 7. auto encoder
config.ae = edict()
config.ae.input = None
config.ae.input_image_type = None
config.ae.keep_slices = 10000
config.ae.n_model = 10000

config.ae.label_percent = 30

config.ae.do_scaling = True
config.ae.p_scale = 0.8
config.ae.do_rotation = True
config.ae.p_rot = 0.8
config.ae.do_translate = True
config.ae.p_trans = 0.8
config.ae.trans_max_shifts = 10

config.ae.gt_opening = False
config.ae.gt_opening_iter = 10
config.ae.extra_near_nums = ''
config.ae.extra_near_dist = ''
config.ae.extra_far_nums = ''
config.ae.extra_radius = ''
config.ae.extra_slices = ''

config.ae.dilate_nums = ''
config.ae.dilate_iters = ''
config.ae.erode_nums = ''
config.ae.erode_dlices = ''
config.ae.erode_iters = ''
config.ae.discard_nums = ''
config.ae.discard_slices = ''
config.ae.extend_nums_head = ''
config.ae.extend_nums_branch = ''
config.ae.close_iters = ''

# 8. gnn
config.gnn = edict()
config.gnn.gnn_type = None
config.gnn.n_blocks = 0
config.gnn.resolution = 0  # 24, 12, 6 for promise prostate, to be refined
config.gnn.sub_sample = False
config.gnn.bn_layer = False



# 9. shape
config.shape = edict()
config.shape.source = None    # 'in_dataset'
config.shape.n_model = 10000  # the same as ae.n_model
config.shape.only_full = False  # only use one full mask for training


# 10. superpixel
config.sp = edict()
config.sp.use_type = ''   # 'decoder', 'gnn'
config.sp.dir = ''
config.sp.mask_ratios = ''
config.sp.loss_weight = 1.0
config.sp.finetune = ''
config.sp.loss = ''


# update method
def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v


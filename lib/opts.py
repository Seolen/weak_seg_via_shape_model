import argparse
import os


def get_args():
    project_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser(description='3D Semantic Object Part Detection')

    parser.add_argument('--demo',          default=None,                     help='Train/val/test split in testing mode')
    parser.add_argument('--parallel',      action='store_true',              help='Data parallel, true if using multi GPUs')
    parser.add_argument('--snapshot',      default=5,       type=int,        help='Model checkpoint')
    parser.add_argument('--n_epochs',      default=1,       type=int,        help='Number of training epochs')
    parser.add_argument('--train_batch',   default=1,       type=int,        help='Training mini-batch size')
    parser.add_argument('--valid_batch',   default=1,       type=int,        help='Validation mini-batch size')
    parser.add_argument('--test_batch',    default=1,       type=int,        help='Test mini-batch size')
    parser.add_argument('--num_workers',   default=4,       type=int,        help='')

    parser.add_argument('--lr',            default=1e-2,    type=float,      help='Learning rate')
    parser.add_argument('--lr_update',     default=1e-5,    type=float,      help='Learning rate after update')
    parser.add_argument('--lr_decay',      default='constant', type=str,     help='Learning decay policy: {constant, multistep, poly}')
    parser.add_argument('--momentum',      default=0.9,     type=float,      help='Momentum')
    parser.add_argument('--weight_decay',  default=1e-4,    type=float,      help='Weight decay')
    parser.add_argument('--step_size',     default=None,    type=int,        help='Learning rate decay step size')
    parser.add_argument('--gamma',         default=0.1,     type=float,      help='Learning rate decay gamma')

    parser.add_argument('--data_dir',      default=None,                     help='Data directory')
    parser.add_argument('--model_dir',     default=None,                     help='Model directory')
    parser.add_argument('--log_dir',       default=None,                     help='Log directory')
    parser.add_argument('--tb_dir',        default=None,                     help='Tensorboard file directory')
    parser.add_argument('--pre_model',     default=None,    type=int,        help='Pretrained model name')
    parser.add_argument('--vgg',           default=None,                     help='Pretrained VGG model')

    parser.add_argument('--id',            default=None,                     help='Experiment ID')
    parser.add_argument('--use_weak',      default=0,       type=int,        help='load weak or fully data loader')
    parser.add_argument('--scribble',      default=0,       type=int,        help='{0: use bbox label, 1: use scribble label}')
    parser.add_argument('--fg_weight',     default=8,       type=float,      help='class weight for CEloss: weight=(1, fg_weight)')
    parser.add_argument('--loss_type',     default=0,       type=int,        help='loss types: {0: weighted CE, 1: WCE+size_constraint, 2: WCE+size+bg_constraint}')
    parser.add_argument('--ce_type',       default='ce',    type=str,        help='ce loss: {ce: nn.CrossEntropyLoss, focal: FocalLoss }')
    parser.add_argument('--alpha',         default=0,       type=float,      help='balance weight of pce and bg+size loss')
    parser.add_argument('--beta',           default=0,      type=float,      help='balance weight of bg_constraint and size_constraint')
    parser.add_argument('--use_tight_size', default=0,      type=int,        help='size constraint: employ customized size for each volume based on box')

    parser.add_argument('--shuffle',       action='store_true',              help='Shuffle data')
    parser.add_argument('--input_res',     default=256,     type=int,        help='Input image resolution')
    parser.add_argument('--output_res',    default=64,      type=int,        help='Output map resolution')
    parser.add_argument('--l_alpha',       default=1,       type=float,      help='Loss weight alpha')
    parser.add_argument('--l_beta',        default=1,       type=float,      help='Loss weight beta')

    args = parser.parse_args()

    args.data_dir = os.path.join(project_dir, '../data')
    args.model_dir = os.path.join(project_dir, '../checkpoints', args.id)
    args.log_dir = os.path.join(project_dir, '../log')
    args.tb_dir = os.path.join(args.log_dir, 'tb_' + args.id)
    if args.pre_model is None:
        args.pre_model = 'best_model.pth'
    else:
        args.pre_model = 'model_epoch_{}.pth'.format(args.pre_model)
    if args.vgg is not None:
        args.vgg = 'vgg16.pth'

    return args


def set_opt(value):
    global opt
    opt = value


def get_opt():
    return opt

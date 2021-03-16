import os
from lib.configs.parse_arg import opt, args
from lib.utils import random_init
from lib.wsss import WSSS


if __name__ == '__main__':
    # make default dirs
    random_init(args.seed)
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    # run
    wsss_seg = WSSS()
    if args.demo == '':
        wsss_seg.train()
    else:
        # wsss_seg.test()
        wsss_seg.vis()
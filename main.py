import yaml
import csv
import argparse
import os
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from utils.util import set_seed
import warnings
warnings.filterwarnings("ignore")


def main(config, args):
    set_seed(config.TRAIN.manualSeed)
    Mission = TextSR(config, args)
    if args.test:
        Mission.test()
    else:
        log_path = os.path.join(config.TRAIN.ckpt_dir, "log.csv")
        if not os.path.exists(log_path):
            with open(log_path, "w+") as out:
                writer = csv.writer(out)
                writer.writerow(["epoch", "dataset", "accuracy", "psnr_avg", "ssim_avg", "best", "best_sum"])
        Mission.train() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--pre_training', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='/root/dataset/TextZoom/test/medium', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='')
    parser.add_argument('--prior_dim', type=int, default=1024, help='')
    parser.add_argument('--dec_num_heads', type=int, default=16, help='')
    parser.add_argument('--dec_mlp_ratio', type=int, default=4, help='')
    parser.add_argument('--dec_depth', type=int, default=1, help='')
    parser.add_argument('--max_gen_perms', type=int, default=1, help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--perm_forward', action='store_true', default=False, help='')
    parser.add_argument('--perm_mirrored', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)

import torch
from torch import cos, sin

import os
import logging
import argparse
import random
from time import time
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build and train DeepSorption.")

    parser.add_argument('--pretrain', type=str, default='', help='Whether to load the pretrained model weights.')
    parser.add_argument('--atom_class', type=int, default=100, help='The default number of atom classes + 1.')
    parser.add_argument('--n_encoder', type=int, default=6, help='Number of stacked encoder.')
    parser.add_argument('--embed_dim', type=int, default=512, help='Dimension of PE, embed_dim % head == 0.')
    parser.add_argument('--fnn_dim', type=int, default=2048, help='Dimension of FNN.')
    parser.add_argument('--head', type=int, default=8, help='Number of heads in multi-head attention.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum length for the positional embedding layer.')
    parser.add_argument('--dist_bar', type=list, default=[[5, 8, 12, 1e10]], help='Dist bar.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=300, help='Number of epoch.')
    parser.add_argument('--bs', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch.')

    parser.add_argument('--gpu_id', type=str, default='0', help='Index for GPU')
    parser.add_argument('--save_path', default='save/', help='Path to save the model and the logger.')
    parser.add_argument('--cpt_name', default='../save/coremof/Coremof_Tgt_7_16.pt', help='Path to load the model and predict.')
    parser.add_argument('--expmof', default='c2h2', choices=['c2h2', 'co2'])
    return parser.parse_args()


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    level_relations = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                       'error': logging.ERROR, 'crit': logging.CRITICAL}  

    def __init__(self, path, filename, level='info'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.logger = logging.getLogger(path + filename)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        self.logger.addHandler(sh)

        th = logging.FileHandler(path + filename, encoding='utf-8')
        self.logger.addHandler(th)





if __name__ == '__main__':
    print()

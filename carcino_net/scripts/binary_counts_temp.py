"""
To verify the mapping of class label to pixel values.
"""
from typing import Tuple, List
import sys
import argparse
import cv2
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from carcino_net.dataset.utils import file_part
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Mask Label Collation')
parser.add_argument('--mask_dir',
                    default='/tmp/ramdisk/SICAPV2Fixed/masks_binary/',
                    help='path to masks')
parser.add_argument('--mask_ext',
                    default='.png',
                    help='mask ext')
parser.add_argument('--export_folder',
                    type=str,
                    default='./data/',
                    help='where to export')
opt, _ = parser.parse_known_args(argv)

if __name__ == '__main__':
    mask_list = glob.glob(os.path.join(opt.mask_dir, f'*{opt.mask_ext}'))

    os.makedirs(opt.export_folder, exist_ok=True)

    pos_count = 0
    neg_count = 0
    for mask_file in tqdm(mask_list):
        mask = cv2.imread(mask_file)
        mask = mask[:, :, 0]
        mask = mask == 255
        pos_count += mask.sum()
        neg_count += mask.size

    with open(os.path.join(opt.export_folder, 'pixel_binary_weight.json'), 'w') as root:
        result = dict()
        result['pos_count'] = int(pos_count)
        result['neg_count'] = int(neg_count)
        result['pos_neg_ratio'] = result['pos_count'] / result['neg_count']
        json.dump(result, root, indent=4)

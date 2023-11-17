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
                    default='/tmp/ramdisk/SICAPV2Fixed/masks_mul/',
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

    count_dict = dict()
    for mask_file in tqdm(mask_list):
        mask = cv2.imread(mask_file, flags=cv2.IMREAD_GRAYSCALE)
        pixels, counts = np.unique(mask, return_counts=True)
        for p, c in zip(pixels.tolist(), counts.tolist()):
            if p not in {0, 1, 2}:
                print(mask_file)
            key = int(p)
            value = int(c)
            count_dict[key] = count_dict.get(key, 0)
            count_dict[key] += value
    total_pixel = sum(count_dict.values())
    max_count = max(count_dict.values())
    with open(os.path.join(opt.export_folder, 'pixel_weight.json'), 'w') as root:
        result = dict()
        result['pixel_count'] = count_dict
        result['pixel_ratio'] = {k: v/total_pixel for k, v in count_dict.items()}
        result['pixel_weight_invert'] = {k: max_count / v for k, v in count_dict.items()}
        sum_weight = sum(result['pixel_weight_invert'].values())
        result['pixel_weight_invert'] = {k: v / sum_weight for k, v in result['pixel_weight_invert'].items()}
        json.dump(result, root, indent=4)

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
from tqdm import tqdm
from carcino_net.dataset.utils import filter_pixel_values, top_k_pixel_values

argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Mask Label Collation')
parser.add_argument('--mask_dir',
                    default='W:/SICAPV2/masks',
                    help='path to masks')
parser.add_argument('--mask_ext',
                    default='.jpg',
                    help='mask ext')
parser.add_argument('--top_k',
                    default=4,
                    type=int,
                    help='max num of classes per image')
parser.add_argument('--area_thresh',
                    type=float,
                    default=0.1,
                    help='max num of classes per image')
parser.add_argument('--gleason_dict',
                    type=str,
                    default='./data/gleason.json',
                    help='where to export')
parser.add_argument('--export_folder',
                    type=str,
                    default='Z:/UTAH/SICAPV2/mask_curated',
                    help='where to export')
opt, _ = parser.parse_known_args(argv)


if __name__ == '__main__':
    os.makedirs(opt.export_folder, exist_ok=True)

    with open(opt.gleason_dict, 'r') as root:
        gleason_dict = json.load(root)
    gleason_group = gleason_dict['group']
    gleason_group = {int(k): int(v) for k, v in gleason_group.items()}
    pixel_list = list(gleason_group.keys())

    mask_list = glob.glob(os.path.join(opt.mask_dir, f'*{opt.mask_ext}'))
    for mask_file in tqdm(mask_list):
        mask = cv2.imread(mask_file)
        mask = filter_pixel_values(mask, pixel_list)
        pixel_count: List[Tuple[int, int]] = top_k_pixel_values(mask, opt.top_k)
        pixel_count_filtered = [x for x in pixel_count if x[1] / mask.size >= opt.area_thresh]
        pixel_local = [x[0] for x in pixel_count_filtered]
        mask = filter_pixel_values(mask, pixel_local)

        for pix in pixel_local:
            # this is under assumption that all pixel values are larger than group num
            mask[mask == pix] = gleason_group[pix]

        fpart = os.path.splitext(os.path.basename(mask_file))[0]
        dest = os.path.join(opt.export_folder, f"{fpart}.png")
        cv2.imwrite(dest, mask)

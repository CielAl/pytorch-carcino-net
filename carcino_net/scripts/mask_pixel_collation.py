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
from carcino_net.dataset.utils import top_k_pixel_values, gleason_score_to_grade_group
argv = sys.argv[1:]
parser = argparse.ArgumentParser(description='Mask Label Collation')
parser.add_argument('--mask_dir',
                    default='W:/SICAPV2_fix/masks',
                    help='path to masks')
parser.add_argument('--mask_ext',
                    default='.png',
                    help='mask ext')
parser.add_argument('--top_k',
                    default=4,
                    type=int,
                    help='max num of classes per image')
parser.add_argument('--area_thresh',
                    type=float,
                    default=0.1,
                    help='max num of classes per image')
parser.add_argument('--export_folder',
                    type=str,
                    default='./data/',
                    help='where to export')
opt, _ = parser.parse_known_args(argv)

if __name__ == '__main__':
    mask_list = glob.glob(os.path.join(opt.mask_dir, f'*{opt.mask_ext}'))

    pixel_set = set()
    count_map = dict()
    for mask_file in tqdm(mask_list):
        mask = cv2.imread(mask_file)
        pixel_count: List[Tuple[int, int]] = top_k_pixel_values(mask, opt.top_k)
        pixel_count_filtered = [x for x in pixel_count if x[1] / mask.size >= opt.area_thresh]
        pixel_class = {x[0] for x in pixel_count_filtered}
        pixel_set = pixel_set.union(pixel_class)

        for pix, count in pixel_count_filtered:
            count_map[pix] = count_map.get(pix, set())
            count_map[pix].add(count)

    os.makedirs(opt.export_folder, exist_ok=True)
    result = dict()
    with open(os.path.join(opt.export_folder, 'pixel_collate.json'), 'w') as root:
        result['pixel_set'] = list(pixel_set)
        result['pixel_set'].sort()
        result['pixel_set'] = [int(x) for x in result['pixel_set']]
        result['count_map_max'] = {int(k): int(max(v)) for k, v in count_map.items()}
        json.dump(result, root, indent=4)
    result['pixel_set'].sort()

    gleason_out = dict()
    with open(os.path.join(opt.export_folder, 'gleason.json'), 'w') as root:

        all_scores = list(range(0, 11))
        gleason_score_dict = {pix: score for pix, score in zip(result['pixel_set'], all_scores)}
        gleason_out['score'] = gleason_score_dict
        gleason_group_dict = {pix: gleason_score_to_grade_group(gleason_score_dict[pix])
                              for pix in result['pixel_set']}
        gleason_out['group'] = gleason_group_dict
        json.dump(gleason_out, root, indent=4)
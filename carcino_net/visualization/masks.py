from typing import List, Optional, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from skimage import img_as_ubyte


def export_plots(images: List[np.ndarray], titles: List[str], dest_name: Optional[str] = None):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 4), dpi=300)
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])
    for i, ax in enumerate(axs):
        img_uint8 = img_as_ubyte(images[i])
        cmap = None if img_uint8.ndim >= 3 else 'gray'
        ax.imshow(img_uint8, cmap, vmin=0, vmax=255)
        ax.set_title(titles[i])
        ax.axis('off')
    if dest_name is not None:
        fig.savefig(dest_name, bbox_inches='tight')
        plt.clf()


def export_showcase(image: np.ndarray,
                    ground_truth_mask: np.ndarray, pred_mask: np.ndarray, dest_name: Optional[str] = None):
    images = [image, ground_truth_mask, pred_mask]
    titles = ['input', 'ground truth', 'prediction']
    export_plots(images, titles, dest_name)


def pred_to_label(prob_mask: np.ndarray, class_axis: int = -1):
    return prob_mask.argmax(axis=class_axis)


def to_instance_map_helper(label_mask: np.ndarray, cmap_func: Callable):
    return cmap_func(label_mask)[..., 0:3]


def to_instance_map(label_mask: np.ndarray, cmap: str = 'tab20'):
    cmap_func = get_cmap(cmap)
    return to_instance_map_helper(label_mask, cmap_func=cmap_func)

import os

import numpy as np
from typing import List, Sequence


def top_k_pixel_values(image: np.ndarray, k: int):
    flat_image = image.ravel()
    values, counts = np.unique(flat_image, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    top_values = values[sorted_indices[:k]]
    top_counts = counts[sorted_indices[:k]]
    return list(zip(top_values, top_counts))


def clean_mask(compressed_mask: np.ndarray, class_labels: List[int]):

    # Initialize an empty mask of the same shape as the compressed mask.
    cleaned_mask = np.zeros_like(compressed_mask)

    # For each class label, compute a distance map.
    # For 3D masks (e.g., RGB images), we consider the Euclidean distance in color space.
    distance_maps = [np.linalg.norm(compressed_mask - label, axis=-1) for label in class_labels]

    # For each pixel, find the class label with the minimum distance.
    min_distance_index = np.argmin(distance_maps, axis=0)

    # For RGB masks, assign the corresponding RGB value. For grayscale, assign the scalar value.
    for i, label in enumerate(class_labels):
        cleaned_mask[min_distance_index == i] = label

    return cleaned_mask


def gleason_score_to_grade_group(score: int):
    if score == 0:
        return 0

    if 0 < score <= 6:
        return 1

    if score == 7:
        return 2

    if score == 8:
        return 3

    if score > 8:
        return 3
    raise ValueError(f"Invalid score: {score}")


def filter_pixel_values(image: np.ndarray, valid_labels: List[int]):
    mask = np.isin(image, valid_labels)
    image[~mask] = 0
    return image


def file_part(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]


def mask_pixel_map(mask: np.ndarray, pixels_to_map: Sequence, pixel_map: dict) -> np.ndarray:
    # todo check if the map routes is acyclic
    for pix in pixels_to_map:
        # this is under assumption that all pixel values are larger than group num
        mask[mask == pix] = pixel_map[pix]
    return mask

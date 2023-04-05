from functools import partial
from typing import Callable, Dict, List, Optional

import albumentations as A
import numpy as np
from loguru import logger
from rio_color.operations import gamma, sigmoidal


def normalize_func(img: np.array, means: np.array, stdevs: np.array) -> np.array:
    img = (img - means) / stdevs
    return img


def build_normalizing_transform(
    normalizing_params: Optional[Dict[str, List[float]]],
    is_inverse: bool = False,
) -> Callable:
    """
    Set up a normalizing transform based on the mean and standard deviation

    Optionally return the inverse transofrm
    """

    if normalizing_params is None:
        # parameters for resnet trained on imagenet
        normalizing_params = {
            "means": [0.485, 0.456, 0.406],
            "stdevs": [0.229, 0.224, 0.225],
        }

    means = np.array(normalizing_params["means"]).reshape(-1, 1, 1)
    stdevs = np.array(normalizing_params["stdevs"]).reshape(-1, 1, 1)

    assert means.shape == stdevs.shape

    if is_inverse:
        means = -means / stdevs
        stdevs = 1 / stdevs

    normalizing_transform = partial(normalize_func, means=means, stdevs=stdevs)

    return normalizing_transform


def build_augmentation_transforms(
    augmentors: Optional[List[str]],
) -> Optional[A.Compose]:
    if augmentors is None:
        return None

    augmentors_dict = {
        "Blur": A.Blur(),
        "RandomRotate90": A.RandomRotate90(),
        "HorizontalFlip": A.HorizontalFlip(),
        "VerticalFlip": A.VerticalFlip(),
        "GaussianBlur": A.GaussianBlur(),
        "GaussNoise": A.GaussNoise(),
        "RGBShift": A.RGBShift(),
        "ToGray": A.ToGray(),
    }

    aug_transforms = []
    for augmentor in augmentors:
        try:
            aug_transforms.append(augmentors_dict[augmentor])
        except KeyError as k:
            logger.warning(
                f"{k} is an unknown augmentor. Continuing without {k}. "
                f"Known augmentors are: {list(augmentors_dict.keys())}"
            )
    aug_transforms = A.Compose(aug_transforms)

    return aug_transforms


def color_correct_image(img: np.array) -> np.array:
    img = sigmoidal(img, 3, 0.5)
    img = gamma(img, 1.5)

    return img

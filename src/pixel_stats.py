from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
from loguru import logger


def parallel_variance(mean_a, count_a, var_a, mean_b, count_b, var_b):
    """Compute the variance based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Lifted from rastervision
    https://docs.rastervision.io/en/0.20/_modules/rastervision/core/raster_stats.html#RasterStats.compute

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        var_a: the variance of partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b
        var_b: the variance of partition b

    Return:
        the variance of the two partitions if they were combined
    """
    delta = mean_b - mean_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta**2 * count_a * count_b / (count_a + count_b)
    var = M2 / (count_a + count_b - 1)
    return var


def parallel_mean(mean_a, count_a, mean_b, count_b):
    """Compute the mean based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b

    Return:
        the mean of the two partitions if they were combined
    """
    mean = (count_a * mean_a + count_b * mean_b) / (count_a + count_b)
    return mean


def get_mean_and_std_pixel_vals(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]]
) -> Tuple[np.array, np.array]:
    """Gets the mean and standard deviation across the entire dataset per channel
    Inspired from rastervision
    https://docs.rastervision.io/en/0.20/_modules/rastervision/core/raster_stats.html#RasterStats.compute
    """
    # For each tile, compute the mean and var of that tile and then update the
    # running mean and var.
    running_count = 0
    running_means = 0
    running_vars = 0

    for stream_unit in img_mask_stream:
        img = stream_unit["img"]
        channel_means = np.nanmean(img, axis=(1, 2))
        channel_vars = np.nanvar(img, axis=(1, 2))

        non_null_pixels = ~np.isnan(img[0, :, :])
        pixel_count = np.sum(non_null_pixels)

        running_vars = parallel_variance(
            channel_means,
            pixel_count,
            channel_vars,
            running_means,
            running_count,
            running_vars,
        )
        running_means = parallel_mean(
            channel_means, pixel_count, running_means, running_count
        )
        running_count += pixel_count

    overall_means = running_means
    overall_stds = np.sqrt(running_vars)

    return overall_means, overall_stds


def get_min_max_pixel_values(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]]
) -> Tuple[int, int]:
    """Get the min and max pixel values across the images (for scaling)"""

    max_val = None
    min_val = None

    for stream_unit in img_mask_stream:
        img = stream_unit["img"]
        img_max = np.max(img)
        img_min = np.min(img)

        if max_val is None:
            max_val = img_max
        elif max_val < img_max:
            max_val = img_max

        if min_val is None:
            min_val = img_min
        elif min_val > img_min:
            min_val = img_min

    return max_val, min_val


def get_img_unique_shapes(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]]
) -> Dict[str, Dict[Tuple, int]]:
    """Get the unique shapes of the images along with each respective count"""

    unique_shapes_dict = {"imgs": {}, "masks": {}}

    for stream_unit in img_mask_stream:
        img_shape = stream_unit["img"].shape

        if img_shape not in unique_shapes_dict["imgs"].keys():
            unique_shapes_dict["imgs"][img_shape] = 1
        else:
            unique_shapes_dict["imgs"][img_shape] += 1

        if stream_unit["mask"] is not None:
            mask_shape = stream_unit["mask"].shape

            if mask_shape not in unique_shapes_dict["masks"].keys():
                unique_shapes_dict["masks"][mask_shape] = 1
            else:
                unique_shapes_dict["masks"][mask_shape] += 1

    return unique_shapes_dict


def get_percentile_pixel_values(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]],
    percentile: int = 50,
) -> Dict[str, float]:
    """
    Gets the list of percentile pixel values of all images

    This is used as a diagnostic to find cloudy images
    """

    percentile_vals = {}

    for stream_unit in img_mask_stream:
        img = stream_unit["img"]
        data_id = stream_unit["data_id"]

        percentile_val = np.percentile(img, q=percentile)
        percentile_vals[data_id] = percentile_val

    return percentile_vals


def get_null_data_ids(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]],
    null_pixel_val: Union[int, float] = 0,
) -> List[str]:
    """Gets the IDs of images that have all null pixels"""

    null_data_ids = []

    for stream_unit in img_mask_stream:
        img = stream_unit["img"]
        data_id = stream_unit["data_id"]

        if (img == null_pixel_val).all():
            null_data_ids.append(data_id)

    return null_data_ids


def validate_image_dims_for_segmentation(
    img_mask_stream: Iterator[Dict[str, Union[np.array, str]]]
) -> None:
    for stream_unit in img_mask_stream:
        img = stream_unit["img"]
        data_id = stream_unit["data_id"]

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = img.shape[1:]
        if not (h % 32 == 0 and w % 32 == 0):
            raise ValueError(f"Incorrect dimensions for image with id {data_id}")

    logger.info("All images successfully validated")

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from lightning import LightningDataModule
from loguru import logger
from matplotlib import colors
from matplotlib.patches import Patch
from rasterio.plot import show, show_hist
from shapely.geometry import Polygon, box
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.image_transform_utils import (
    build_augmentation_transforms,
    build_normalizing_transform,
)


class PondDataset(Dataset):
    """
    Pytorch Dataset to read in satellite images and/or raster masks
    """

    def __init__(
        self,
        imgs_root: Union[str, Path],
        masks_root: Optional[Union[str, Path]] = None,
        mask_fname_suffix: Optional[str] = None,
        nn_normalizing_params: Optional[Dict[str, List[float]]] = None,
        rescale_image_method: str = "normalize_clip_and_rescale",
        augmentors: Optional[List[str]] = None,
        cloud_cover_lookup_fpath: Optional[Union[str, Path]] = None,
    ) -> None:
        self.imgs_root = Path(imgs_root)
        self.img_fname_suffix = ".tif"

        self.masks_root = masks_root
        if self.masks_root is not None:
            self.masks_root = Path(self.masks_root)

        self.mask_fname_suffix = mask_fname_suffix
        if self.mask_fname_suffix is None:
            self.mask_fname_suffix = "-mask.tif"

        self.cloud_cover_tile_lookup = self._get_cloud_cover_tile_lookup(
            cloud_cover_lookup_fpath
        )

        # labels per class
        self.label_mapping = {
            0: "background",
            1: "abandoned",
            2: "extensive",
            3: "intensive",
            4: "censored",
        }

        self.NUM_CLASSES = len(self.label_mapping)

        self.BACKGROUND_PIXEL_VAL = 0

        # color map for plotting
        self.label_cmap = {
            "censored": "black",
            "background": "azure",
            "abandoned": "tab:red",
            "extensive": "tab:green",
            "intensive": "tab:blue",
        }

        # get img ids from the filenames
        img_fnames = os.listdir(self.imgs_root)
        data_ids = [fname.replace(self.img_fname_suffix, "") for fname in img_fnames]

        data_ids = sorted(data_ids)
        self.data_ids = data_ids

        if self.cloud_cover_tile_lookup is not None:
            remove_data_ids = self.cloud_cover_tile_lookup["remove_tile"]
            logger.info(
                f"Removing {len(remove_data_ids):,} tiles based on the cloud cover lookup"
            )
            self.data_ids = [id for id in data_ids if id not in remove_data_ids]

        self.BAND_INDICES = [3, 2, 1]
        # self.BAND_INDICES = [3, 2, 1, 4]
        self.NUM_IN_CHANNELS = len(self.BAND_INDICES)
        self.RGB_INDICES = [3, 2, 1]

        self.NODATA_VAL = 0

        self.RESCALE_IMAGE_METHOD = rescale_image_method
        # this are precomputed statistics based on the training set
        # these are hardcoded for now
        # you can use the get_mean_and_std_pixel_vals method to compute these
        # this normalization is for controlling the variance
        self.DATA_NORMALIZING_PARAMS = {
            "means": [459.06415365, 686.84029586, 568.45791792],
            "stdevs": [221.29763659, 249.20891633, 295.51260358],
        }

        self.data_normalize = build_normalizing_transform(self.DATA_NORMALIZING_PARAMS)

        # this is a normalization based on the statistics of the images that the
        # neural net was pretrained on
        self.nn_normalize = build_normalizing_transform(nn_normalizing_params)
        self.inv_nn_normalize = build_normalizing_transform(
            nn_normalizing_params,
            is_inverse=True,
        )

        self.aug_transforms = build_augmentation_transforms(augmentors)

        # set CRS
        self.crs = self._get_crs()

    def _get_crs(self) -> rio.CRS:
        """
        Get the CRS based on the first image's CRS

        Underlying assumping is that all tiles have the same CRS
        """
        data_id = self.data_ids[0]
        img_fpath, _ = self.get_img_and_mask_paths(data_id)
        with rio.open(img_fpath) as src:
            crs = src.crs

        return crs

    @staticmethod
    def _get_cloud_cover_tile_lookup(
        cloud_cover_lookup_fpath: Optional[Union[str, Path]]
    ) -> Optional[Dict[str, List[str]]]:
        if cloud_cover_lookup_fpath is not None:
            cloud_cover_lookup = pd.read_csv(cloud_cover_lookup_fpath, dtype=str)

            bool_mask = cloud_cover_lookup["tile_decision"] == "remove_tile"
            remove_tile_keys = cloud_cover_lookup.loc[bool_mask, "quadkey"].tolist()

            bool_mask = cloud_cover_lookup["tile_decision"] == "make_all_background"
            make_background_keys = cloud_cover_lookup.loc[bool_mask, "quadkey"].tolist()

            cloud_cover_tile_lookup = {
                "remove_tile": remove_tile_keys,
                "make_background": make_background_keys,
            }
        else:
            cloud_cover_tile_lookup = None

        return cloud_cover_tile_lookup

    def _aug_transform(
        self, image: np.array, mask: Optional[np.array]
    ) -> Tuple[np.array, Optional[np.array]]:
        """
        Convenience function for using aug_transforms with and without a mask
        """

        ## channels should be last when doing augmentation (i.e. H x W x C)
        image = np.moveaxis(image, source=0, destination=-1)

        if mask is None:
            aug_transformed = self.aug_transforms(image=image)
            image = aug_transformed["image"]
        else:
            aug_transformed = self.aug_transforms(image=image, mask=mask)
            image = aug_transformed["image"]
            mask = aug_transformed["mask"]

        ## bring channels back to first to follow Pytorch format (i.e. C x W x H)
        image = np.moveaxis(image, source=-1, destination=0)

        return image, mask

    def _background_mask_check(self, data_id: str, mask: np.array) -> np.array:
        """
        Force tiles to be background if necessary
        """
        output_mask = mask
        if self.cloud_cover_tile_lookup is not None:
            force_background_data_ids = self.cloud_cover_tile_lookup["make_background"]
            if data_id in force_background_data_ids:
                output_mask = np.full(
                    shape=mask.shape,
                    fill_value=self.BACKGROUND_PIXEL_VAL,
                    dtype=np.int64,
                )

        return output_mask

    def _rescale_image(
        self,
        img: torch.Tensor,
        rescale_image_method: str = "normalize_clip_and_rescale",
    ) -> torch.Tensor:
        """
        Preprocesses images to have values between 0 and 1

        Option 1: MinMax Scaling (rescale_only)
        Option 2: Normalize, Clip, and MinMaxScale (normalize_clip_and_rescale)
        """

        assert rescale_image_method in ["rescale_only", "normalize_clip_and_rescale"]

        if rescale_image_method == "rescale_only":
            # this is based on the max values for the RGB bands
            # https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_americas#bands
            # can also verify values using the find_min_max_pixel_values function
            MAX_IMG_VAL = 10_000

            img = img / MAX_IMG_VAL
        if rescale_image_method == "normalize_clip_and_rescale":
            # standard dev of 3 is a reasonable value
            MAX_STDEV = 3

            # normalize image
            img = self.data_normalize(img)
            # clip values that are these many standard deviations away
            img = np.clip(a=img, a_min=-MAX_STDEV, a_max=MAX_STDEV)
            # rescale values to be within 0 and 1
            img = (img + MAX_STDEV) / (2 * MAX_STDEV)

        return img

    def __len__(self) -> int:
        n_data = len(self.data_ids)
        return n_data

    def __repr__(self) -> str:
        n_data = len(self.data_ids)

        if self.masks_root is not None:
            out = f"""
            Pointing to {n_data:,} images
            image directory: "{self.imgs_root}"
            mask directory: "{self.masks_root}"
            """
        else:
            out = f"""
            Pointing to {n_data:,} images
            image directory: "{self.imgs_root}"
            """
        return out

    def __getitem__(
        self, idx: int, use_aug_transforms: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        use_aug_transforms is a flag so you can turn off data augmentation for the validation set
        """

        data_id = self.data_ids[idx]
        img_fpath, mask_fpath = self.get_img_and_mask_paths(data_id)

        with rio.open(img_fpath) as src:
            img = src.read(indexes=self.BAND_INDICES)
            img = np.array(img, dtype=np.float32)

        if mask_fpath is not None:
            with rio.open(mask_fpath) as src:
                mask = src.read(indexes=1)
                mask = np.array(mask, dtype=np.int64)
                mask = self._background_mask_check(data_id, mask)
        else:
            mask = None

        if self.aug_transforms is not None and use_aug_transforms:
            img, mask = self._aug_transform(image=img, mask=mask)

        img = self._rescale_image(img, rescale_image_method=self.RESCALE_IMAGE_METHOD)

        img = self.nn_normalize(img)

        sample = {}
        sample["image"] = torch.from_numpy(img).float()
        sample["image_idx"] = torch.tensor(idx)
        if mask_fpath is not None:
            sample["mask"] = torch.from_numpy(mask)

        return sample

    def img_mask_stream(self) -> Iterator[Dict[str, Any]]:
        """
        Makes a generator of image and mask pairs
        """

        for data_id in tqdm(self.data_ids):
            img_fpath, mask_fpath = self.get_img_and_mask_paths(data_id)

            with rio.open(img_fpath) as src:
                img = src.read(indexes=self.BAND_INDICES)

            if mask_fpath is not None:
                with rio.open(mask_fpath) as src:
                    mask = src.read(indexes=1)
            else:
                mask = None

            output = {
                "data_id": data_id,
                "img": img,
                "mask": mask,
            }

            yield output

    def check_imgs_masks_same_ids(self) -> None:
        if self.masks_root is None:
            logger.warning(
                "masks_root is not defined. There are no masks to compare to"
            )
            return None

        img_fnames = os.listdir(self.imgs_root)
        mask_fnames = os.listdir(self.masks_root)

        # check if masks have same ids
        img_data_ids = [
            fname.replace(self.img_fname_suffix, "") for fname in img_fnames
        ]
        mask_data_ids = [
            fname.replace(self.mask_fname_suffix, "") for fname in mask_fnames
        ]

        if set(img_data_ids) != set(mask_data_ids):
            logger.warning("IDs of images and masks are mismatched. Please recheck")
        else:
            logger.info("IDs of images and masks are matched.")

    def plot_img_histogram(self, idx: int, figsize: Tuple[int, int] = (5, 4)) -> None:
        """Plot the histogram of the raster"""
        data_id = self.data_ids[idx]
        img_fpath, _ = self.get_img_and_mask_paths(data_id)

        fig, ax = plt.subplots(figsize=figsize)
        with rio.open(img_fpath) as src:
            img = src.read(indexes=self.BAND_INDICES)

            title = f"Histogram for item {idx} (img {data_id})"
            label = ["R", "G", "B", "NIR"]

            show_hist(
                img,
                ax=ax,
                title=title,
                label=label,
                bins=50,
                lw=0.0,
                stacked=False,
                alpha=0.3,
                histtype="stepfilled",
            )

            plt.show()

    def plot_img(
        self,
        idx: int,
        figsize: Tuple[int, int] = (8, 8),
        is_ok_missing_mask: bool = True,
        use_aug_transforms=False,
    ) -> None:
        """
        Plot the color corrected image with its corresponding mask
        if available
        """
        data_id = self.data_ids[idx]
        img_fpath, mask_fpath = self.get_img_and_mask_paths(data_id)

        if not is_ok_missing_mask and mask_fpath is None:
            raise ValueError(f"Data ID {data_id} has no mask file")

        cmap = colors.ListedColormap(
            colors=[self.label_cmap[label] for i, label in self.label_mapping.items()]
        )

        # actual labels should be in between the boundary values
        boundaries = [i for i in self.label_mapping.keys()]
        boundaries = [i - 0.5 for i in boundaries] + [boundaries[-1] + 0.5]
        norm = colors.BoundaryNorm(
            boundaries=boundaries,
            ncolors=len(self.label_mapping),
        )

        with rio.open(img_fpath) as src:
            img = src.read(indexes=self.RGB_INDICES)
            img_raster_transform = src.transform

        if mask_fpath is not None:
            with rio.open(mask_fpath) as mask_src:
                mask = mask_src.read(indexes=1)
                mask = self._background_mask_check(data_id, mask)
                mask_raster_transform = mask_src.transform
            ncols = 2
        else:
            mask = None
            ncols = 1

        if self.aug_transforms is not None and use_aug_transforms:
            img, mask = self._aug_transform(image=img, mask=mask)

        img = self._rescale_image(img, rescale_image_method=self.RESCALE_IMAGE_METHOD)

        fig, ax = plt.subplots(ncols=ncols, figsize=figsize)

        if ncols == 1:
            img_ax = ax
            mask_ax = None
        elif ncols == 2:
            img_ax = ax[0]
            mask_ax = ax[1]
        else:
            raise ValueError(f"Invalid ncols = {ncols}")

        show(img, ax=img_ax, transform=img_raster_transform)
        if mask is not None:
            show(
                mask,
                ax=img_ax,
                transform=mask_raster_transform,
                contour=True,
                colors="black",
                contour_label_kws={},
                linestyles="dashed",
                linewidths=0.6,
            )
            show(
                mask,
                ax=mask_ax,
                transform=mask_raster_transform,
                cmap=cmap,
                norm=norm,
                interpolation="none",
            )
            patches = [
                Patch(color=color, label=label)
                for label, color in self.label_cmap.items()
            ]

            ax[1].legend(
                handles=patches,
                bbox_to_anchor=(1.7, 1),
                loc="upper right",
            )

        img_ax.set_title(f"item {idx} (img {data_id})")
        img_ax.axis("off")

        if mask_ax is not None:
            mask_ax.set_title("mask")
            mask_ax.axis("off")
        plt.show()

    def get_tile_bboxes(self) -> Dict[str, Polygon]:
        """
        Gets the bounding box polygons of all tiles

        Useful if you need the vector representation
        """

        tile_bboxes = {}

        for data_id in self.data_ids:
            img_fpath, _ = self.get_img_and_mask_paths(data_id)

            with rio.open(img_fpath) as src:
                bbox = box(*src.bounds)
                tile_bboxes[data_id] = bbox

        return tile_bboxes

    def plot_img_batch(
        self,
        idxs: List[int],
        n_imgs_per_row: int = 5,
        size: int = 10,
    ) -> None:
        data_ids = [self.data_ids[idx] for idx in idxs]

        ncols = min(len(data_ids), n_imgs_per_row)
        nrows = np.ceil(len(data_ids) / n_imgs_per_row).astype(int)

        figsize = (size, size * (nrows / ncols))

        fig = plt.figure(figsize=figsize)

        ax = []

        for i, idx in enumerate(idxs):
            data_id = self.data_ids[idx]
            img_fpath, _ = self.get_img_and_mask_paths(data_id)
            ax.append(fig.add_subplot(nrows, ncols, i + 1))
            with rio.open(img_fpath) as src:
                img = src.read(indexes=self.RGB_INDICES)

                img = self._rescale_image(
                    img, rescale_image_method=self.RESCALE_IMAGE_METHOD
                )

                show(
                    img,
                    ax=ax[-1],
                    transform=src.transform,
                )
                ax[-1].set_title(f"item {idx}")
                ax[-1].axis("off")

        plt.show()

    def get_img_and_mask_paths(
        self,
        data_id: str,
    ) -> Tuple[Path]:
        """Get paths to the tiff files of the image and the mask if available"""

        img_fname = f"{data_id}{self.img_fname_suffix}"
        img_fpath = self.imgs_root / img_fname

        mask_fname = f"{data_id}{self.mask_fname_suffix}"

        if self.masks_root is None:
            mask_fpath = None
        else:
            mask_fpath = self.masks_root / mask_fname
            if not mask_fpath.exists():
                mask_fpath = None

        return img_fpath, mask_fpath

    def save_predict_mask(
        self,
        idx: int,
        pred_mask: np.array,
        output_dir: Union[str, Path],
    ) -> None:
        """
        Writes to disk the prediction mask as a geoTIFF with the same
        georeferencing as the source image
        """

        data_id = self.data_ids[idx]
        img_fpath, _ = self.get_img_and_mask_paths(data_id)
        height, width = pred_mask.shape

        with rio.open(img_fpath) as src:
            # get metadata
            img_meta = src.meta
            img_height = img_meta["height"]
            img_width = img_meta["width"]

            if img_height != height or img_width != width:
                raise ValueError(
                    "Width or height of pred_mask doesnt match source image"
                )

            # Set profile for output tif
            profile = {
                "driver": "GTiff",
                "count": 1,
                "height": height,
                "width": width,
                "transform": img_meta["transform"],
                "dtype": "uint8",
                "crs": img_meta["crs"],
            }

        # Save to disk
        mask_fname = f"{data_id}{self.mask_fname_suffix}"
        mask_fpath = Path(output_dir) / mask_fname
        with rio.open(mask_fpath, "w", **profile) as output:
            output.write(pred_mask, indexes=1)


class PondDataModule(LightningDataModule):
    def __init__(
        self,
        train_imgs_root: Optional[Union[str, Path]] = None,
        train_masks_root: Optional[Union[str, Path]] = None,
        predict_imgs_root: Optional[Union[str, Path]] = None,
        predict_masks_root: Optional[Union[str, Path]] = None,
        batch_size: int = 4,
        num_workers: int = 1,
        nn_normalizing_params: Optional[Dict[str, List[float]]] = None,
        rescale_image_method: str = "normalize_clip_and_rescale",
        augmentors: Optional[List[str]] = None,
        cloud_cover_lookup_fpath: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
           batch_size: The batch size to use in all created DataLoaders
           num_workers: The number of workers to use in all created DataLoaders
        """

        super().__init__()

        self.train_imgs_root = train_imgs_root
        self.train_masks_root = train_masks_root
        self.predict_imgs_root = predict_imgs_root
        self.predict_masks_root = predict_masks_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nn_normalizing_params = nn_normalizing_params
        self.rescale_image_method = rescale_image_method
        self.augmentors = augmentors
        self.cloud_cover_lookup_fpath = cloud_cover_lookup_fpath

    def setup(
        self,
        stage: str,
        valid_set_pct: float = 0.2,
    ) -> None:
        """
        Create the train/val/test splits based on the original Dataset objects.
        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """

        # stage is either 'fit', 'validate', 'test', or 'predict'

        if stage == "fit":
            if self.train_imgs_root is None or self.train_masks_root is None:
                raise ValueError(
                    "Can't train a model if train_imgs_root is None or train_masks_root is None. Specify the dirs of the training data."
                )

            dataset = PondDataset(
                imgs_root=self.train_imgs_root,
                masks_root=self.train_masks_root,
                nn_normalizing_params=self.nn_normalizing_params,
                rescale_image_method=self.rescale_image_method,
                augmentors=self.augmentors,
                cloud_cover_lookup_fpath=self.cloud_cover_lookup_fpath,
            )

            train_val_split = ShuffleSplit(
                test_size=valid_set_pct,
                n_splits=1,
            )
            train_indices, val_indices = next(train_val_split.split(dataset.data_ids))

            self.train_dataset = Subset(dataset, train_indices, use_aug_transforms=True)
            self.val_dataset = Subset(dataset, val_indices, use_aug_transforms=False)

        if stage == "predict":
            if self.predict_imgs_root is None:
                raise ValueError(
                    "Can't train a model if pred_imgs_root is None. Specify the dir of the inference data."
                )

            pred_mask_fname_suffix = "-pred_mask.tif"
            self.predict_dataset = PondDataset(
                imgs_root=self.predict_imgs_root,
                masks_root=self.predict_masks_root,
                mask_fname_suffix=pred_mask_fname_suffix,
                nn_normalizing_params=self.nn_normalizing_params,
                rescale_image_method=self.rescale_image_method,
                augmentors=None,
            )

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for training.
        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for validation.
        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return a DataLoader for prediction.
        Returns:
            prediction data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    **Modified version of the Subset class in Pytorch**

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        use_aug_transforms (bool): whether or not to apply data augmentation
    """

    def __init__(self, dataset, indices, use_aug_transforms) -> None:
        self.dataset = dataset
        self.indices = indices
        self.use_aug_transforms = use_aug_transforms

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = [self.indices[i] for i in idx]
        else:
            idx = self.indices[idx]

        return self.dataset.__getitem__(idx, self.use_aug_transforms)

    def __len__(self):
        return len(self.indices)

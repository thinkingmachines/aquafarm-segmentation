from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    n_wandb_loggers = 0
    for pl_logger in trainer.loggers:
        if isinstance(pl_logger, WandbLogger):
            wandb_logger = pl_logger
            n_wandb_loggers += 1
    if n_wandb_loggers == 0:
        raise ValueError("No Wandb Loggers found")
    elif n_wandb_loggers > 1:
        raise ValueError("Multiple Wandb Loggers found. There should only be one")

    return wandb_logger


class WandbConfusionMatrix(Callback):
    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Called when the validation ends."""

        conf_mat = pl_module._current_val_conf_mat
        assert conf_mat is not None
        ignore_index = pl_module.ignore_index

        val_dataset = trainer.datamodule.val_dataset.dataset
        label_mapping = val_dataset.label_mapping

        conf_mat = self.format_conf_mat(conf_mat, label_mapping, ignore_index)
        wandb_logger = get_wandb_logger(trainer)

        data_table = wandb.Table(dataframe=conf_mat)
        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }

        num_epoch = pl_module.current_epoch
        string_fields = {"title": f"Validation Confusion Matrix Epoch {num_epoch}"}
        wandb_table = wandb_logger.experiment.plot_table(
            vega_spec_name="wandb/confusion_matrix/v1",
            data_table=data_table,
            fields=fields,
            string_fields=string_fields,
        )
        wandb_logger.experiment.log({"conf_mat": wandb_table})

    def format_conf_mat(
        self,
        conf_mat: Tensor,
        label_mapping: Optional[Dict[int, str]] = None,
        ignore_index: Optional[int] = None,
    ) -> pd.DataFrame:
        conf_mat = pd.DataFrame(conf_mat)
        conf_mat.index.name = "Actual"
        conf_mat.columns.name = "Predicted"

        # Unstack to make tuples of actual,pred,count
        conf_mat = conf_mat.unstack().reset_index(name="nPredictions")
        if ignore_index is not None:
            # remove any row that uses ignore_index
            bool_mask = conf_mat["Predicted"] == ignore_index
            bool_mask = bool_mask | (conf_mat["Actual"] == ignore_index)
            conf_mat = conf_mat.loc[~bool_mask, :]

        if label_mapping is not None:
            conf_mat["Predicted"] = conf_mat["Predicted"].map(label_mapping)
            conf_mat["Actual"] = conf_mat["Actual"].map(label_mapping)

        # compute percentages
        conf_mat["PercentPredictions"] = (
            conf_mat["nPredictions"] * 100 / conf_mat["nPredictions"].sum()
        )

        return conf_mat


class WandbSegmentationMasks(Callback):
    def __init__(
        self,
        target_batch_idx: int = 1,
        n_wandb_images: int = 5,
        max_wandb_logs: int = 20,
    ) -> None:
        self.target_batch_idx = target_batch_idx
        self.n_wandb_images = n_wandb_images

        # limit the number of wandb logs so you don't upload too many images
        self.max_wandb_logs = max_wandb_logs
        self.wandb_log_counter = 0

    @staticmethod
    def _prep_image(image: Tensor, dataset: Dataset) -> Tensor:
        """
        Prepare the image so it is human interpretable
        """
        # undo the normalization based statistics of
        # images that the neural net was pretrained on
        image = dataset.inv_nn_normalize(image)

        if dataset.NEEDS_COLOR_CORRECTION_FOR_PLOTS:
            image = image.cpu().numpy()
            image = dataset.color_correct_image(image)
            image = Tensor(image)

        return image

    @staticmethod
    def _get_wandb_image_with_mask(
        image: Tensor,
        mask: Tensor,
        pred_mask: Tensor,
        caption: str,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> wandb.data_types.Image:
        mask = mask.cpu().numpy().astype(np.int8)
        pred_mask = pred_mask.cpu().numpy().astype(np.int8)

        masks_dict = {
            "prediction": {"mask_data": pred_mask},
            "ground truth": {"mask_data": mask},
        }

        if label_mapping is not None:
            masks_dict["prediction"]["class_labels"] = label_mapping
            masks_dict["ground truth"]["class_labels"] = label_mapping

        wandb_image = wandb.Image(image, caption=caption, masks=masks_dict)

        return wandb_image

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # only grab data from a particular batch
        if batch_idx != self.target_batch_idx:
            return None

        # stop if you've logged too many images
        if self.wandb_log_counter >= self.max_wandb_logs:
            message = f"""
            You've reached the maximum number ({self.max_wandb_logs}) of image uploads.
            Will halt uploading segmentation masks. Increase max_wandb_logs if needed.
            """
            logger.warning(message)
            return None

        image_batch = batch["image"]
        image_idx_batch = batch["image_idx"]
        mask_batch = batch["mask"]
        pred_mask_batch = outputs["pred_mask"]

        wandb_logger = get_wandb_logger(trainer)

        val_dataset = trainer.datamodule.val_dataset.dataset
        label_mapping = val_dataset.label_mapping

        wandb_images = []
        zipped_batch = zip(image_idx_batch, image_batch, mask_batch, pred_mask_batch)
        for i, (image_idx, image, mask, pred_mask) in enumerate(zipped_batch):
            if i >= self.n_wandb_images:
                break

            image_idx = image_idx.item()
            data_id = val_dataset.data_ids[image_idx]
            caption = f"item {image_idx} (img {data_id})"

            image = self._prep_image(image, val_dataset)

            wandb_image = self._get_wandb_image_with_mask(
                image, mask, pred_mask, caption, label_mapping
            )
            wandb_images.append(wandb_image)

        wandb_logger.experiment.log({"prediction_masks": wandb_images})
        self.wandb_log_counter += 1

import warnings
from typing import Any, Dict, Tuple, cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    Dice,
    F1Score,
    JaccardIndex,
    MatthewsCorrCoef,
    MetricCollection,
    Precision,
    Recall,
)

from .model_utils import semantic_seg_model_builder

TORCHVISION_ARCH_NAMES = semantic_seg_model_builder.TORCHVISION_ARCH_NAMES


class SegmentationModel(LightningModule):
    """LightningModule for semantic segmentation of images."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function
            ignore_index: Optional integer class index to ignore in the loss and metrics

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.3
           The *ignore_zeros* parameter was renamed to *ignore_index*.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]

        # during inference, force predictions that output ignore_index to be this value
        self.IGNORE_INDEX_PRED_VAL = kwargs.get("ignore_index_pred_val", 0)

        self._config_model()
        self._config_loss()
        self._config_metrics()

        # set to default of Adam optimizer if unspecified
        self.lr = self.hyperparams.get("learning_rate", 1e-4)
        self.lr_scheduler = self.hyperparams.get("lr_scheduler", None)
        self.lr_scheduler_config = self.hyperparams.get("lr_scheduler_config", None)

        # initializing validation confusion matrix for the callback
        self._current_val_conf_mat = None

    def _config_model(self) -> None:
        """
        Configures the model itself

        Also set a reference to the backbone for backbone finetuning callback
        """

        self.model, self.backbone = semantic_seg_model_builder.create_model(
            arch=self.hyperparams["segmentation_model"],
            encoder_name=self.hyperparams["encoder_name"],
            encoder_weights=self.hyperparams["encoder_weights"],
            in_channels=self.hyperparams["in_channels"],
            num_classes=self.hyperparams["num_classes"],
            return_backbone=True,
        )

        self.backbone = self.model.encoder

    def _config_loss(self) -> None:
        """Configures the loss based on kwargs parameters passed to the constructor."""

        if self.hyperparams["loss"] == "ce":
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_value)
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        elif self.hyperparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                mode="multiclass", ignore_index=self.ignore_index, normalized=True
            )
        elif self.hyperparams["loss"] == "dice":
            self.loss = smp.losses.DiceLoss(
                mode="multiclass", ignore_index=self.ignore_index
            )
        elif self.hyperparams["loss"] == "tversky":
            self.loss = smp.losses.TverskyLoss(
                mode="multiclass", ignore_index=self.ignore_index
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not supported."
            )

    def _config_metrics(self) -> None:
        USE_METRICS_LIST = ["Precision", "Recall", "F1Score", "Dice"]

        metrics_dict = {
            "Accuracy": Accuracy(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                average="micro",
                mdmc_average="global",
            ),
            "JaccardIndex": JaccardIndex(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                average="micro",
            ),
            "Precision": Precision(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                mdmc_average="global",
                average="micro",
            ),
            "Recall": Recall(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                mdmc_average="global",
                average="micro",
            ),
            "F1Score": F1Score(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                mdmc_average="global",
                average="micro",
            ),
            "Dice": Dice(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
                mdmc_average="global",
                average="micro",
            ),
            "MatthewsCorrCoef": MatthewsCorrCoef(
                num_classes=self.hyperparams["num_classes"],
                ignore_index=self.ignore_index,
            ),
        }
        # filter the metrics to those in the list
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in USE_METRICS_LIST}

        self.train_metrics = MetricCollection(metrics_dict, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        self.train_conf_mat = ConfusionMatrix(
            num_classes=self.hyperparams["num_classes"],
            ignore_index=self.ignore_index,
        )
        self.val_conf_mat = self.train_conf_mat.clone()
        self.test_conf_mat = self.train_conf_mat.clone()

    def forward(self, image) -> Tensor:
        """Forward pass of the model.

        Args:
            image: tensor of data to run through the model

        Returns:
            output from the model
        """

        if self.hyperparams["segmentation_model"] in TORCHVISION_ARCH_NAMES:
            output = self.model(image)["out"]
        else:
            output = self.model(image)

        return output

    def _shared_eval_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        image = batch["image"]
        mask = batch["mask"]
        pred_mask_logits = self.forward(image)
        pred_mask = pred_mask_logits.argmax(dim=1)

        loss = self.loss(pred_mask_logits, mask)

        eval_outputs = {
            "pred_mask": pred_mask,
            "loss": loss,
        }

        return eval_outputs

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        Returns:
            training loss
        """

        eval_outputs = self._shared_eval_step(batch, batch_idx)
        loss = eval_outputs["loss"]
        pred_mask = eval_outputs["pred_mask"]
        mask = batch["mask"]

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.train_metrics.update(pred_mask, mask)
        self.train_conf_mat.update(pred_mask, mask)

        return loss

    def on_training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        epoch_metrics = self.train_metrics.compute()
        # self.log_dict(epoch_metrics)
        self.train_metrics.reset()

        conf_mat = self.train_conf_mat.compute()
        self.train_conf_mat.reset()

    def validation_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """

        eval_outputs = self._shared_eval_step(batch, batch_idx)
        loss = eval_outputs["loss"]
        pred_mask = eval_outputs["pred_mask"]
        mask = batch["mask"]

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics.update(pred_mask, mask)
        self.val_conf_mat.update(pred_mask, mask)

        return eval_outputs

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        epoch_metrics = self.val_metrics.compute()
        self.log_dict(epoch_metrics)
        self.val_metrics.reset()

        conf_mat = self.val_conf_mat.compute()
        # save it here so it can be accessed by the callback
        # the PlotWandbConfusionMatrix callback
        self._current_val_conf_mat = conf_mat.cpu()
        self.val_conf_mat.reset()

    def test_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """

        eval_outputs = self._shared_eval_step(batch, batch_idx)
        loss = eval_outputs["loss"]
        pred_mask = eval_outputs["pred_mask"]
        mask = batch["mask"]

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics.update(pred_mask, mask)
        self.test_conf_mat.update(pred_mask, mask)

        return eval_outputs

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        epoch_metrics = self.test_metrics.compute()
        self.log_dict(epoch_metrics)
        self.test_metrics.reset()

        conf_mat = self.test_conf_mat.compute()
        self.test_conf_mat.reset()

    def predict_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Performs model inference

        Args:
           batch: the output of your DataLoader
           batch_idx: the index of this batch
        """

        image = batch["image"]
        image_idx = batch["image_idx"]
        pred_mask = self.forward(image)
        pred_mask = pred_mask.argmax(dim=1)

        # force overwrite any predictions of ignore_index to be this value
        if self.ignore_index is not None:
            pred_mask[pred_mask == self.ignore_index] = self.IGNORE_INDEX_PRED_VAL

        return image_idx, pred_mask

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_config = {"optimizer": optimizer}
        lr_scheduler = self._configure_lr_scheduler(optimizer)
        if lr_scheduler:
            optimizer_config["lr_scheduler"] = lr_scheduler

        return optimizer_config

    def _configure_lr_scheduler(self, optimizer: Optimizer) -> Dict[str, Any]:
        supported_schedulers = ["ReduceLROnPlateau", "OneCycleLR"]

        if self.lr_scheduler_config is None:
            kwargs = {}
        else:
            kwargs = self.lr_scheduler_config

        if self.lr_scheduler is None:
            lr_scheduler = {}
        elif self.lr_scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, **kwargs)
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif self.lr_scheduler == "OneCycleLR":
            if "steps_per_epoch" not in kwargs or "epochs" not in kwargs:
                raise ValueError(
                    f"The {self.lr_scheduler} scheduler needs defined steps_per_epoch and epoch parameters"
                )

            scheduler = OneCycleLR(
                optimizer, max_lr=self.lr, **self.lr_scheduler_config
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
            }
        else:
            raise ValueError(
                f"lr_scheduler {self.lr_scheduler} not supported yet. Supported schedulers are {supported_schedulers}"
            )

        return lr_scheduler

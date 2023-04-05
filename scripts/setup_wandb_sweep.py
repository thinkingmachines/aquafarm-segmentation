import copy
from pathlib import Path
from typing import Any, Dict

import lightning as L
import wandb
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger

from ..src.config_utils import build_kwargs_from_config
from ..src.pond_data import PondDataModule, PondDataset
from ..src.seg_model import SegmentationModel

DATA_PATH = Path("../data")
CONFIG_PATH = Path("../config")
MODELS_PATH = Path("../models")
MODELS_PATH.mkdir(exist_ok=True)

CONFIG_FPATH = CONFIG_PATH / "pond_config.yaml"

kwargs_dict = build_kwargs_from_config(DATA_PATH, CONFIG_FPATH, models_path=None)

parent_dir = kwargs_dict["misc_kwargs"]["parent_dir"]
WANDB_PROJ_NAME = f"sweep_{parent_dir}"

DEFAULT_DATAMODULE_KWARGS = kwargs_dict["datamodule_kwargs"]
DEFAULT_LIGHTNINGMODULE_KWARGS = kwargs_dict["lightningmodule_kwargs"]
DEFAULT_TRAINER_KWARGS = {
    "accelerator": "auto",
    "devices": 1,
    "max_epochs": kwargs_dict["trainer_kwargs"]["num_epochs"],
    "max_time": kwargs_dict["trainer_kwargs"]["train_time"],
    "default_root_dir": MODELS_PATH,
    "callbacks": kwargs_dict["trainer_kwargs"]["callbacks"],
    "precision": kwargs_dict["trainer_kwargs"]["precision"],
}
RANDOM_SEED = kwargs_dict["misc_kwargs"]["random_seed"]
L.seed_everything(seed=RANDOM_SEED, workers=True)

DEFAULT_DATASET_KWARGS = kwargs_dict["dataset_kwargs"]


def setup_datamodule(default_datamodule_kwargs, batch_size) -> PondDataModule:
    datamodule_kwargs = copy.deepcopy(default_datamodule_kwargs)
    # wandb to reassign some LightningDataModule kwargs here
    datamodule_kwargs["batch_size"] = batch_size
    pond_datamodule = PondDataModule(**datamodule_kwargs)

    return pond_datamodule


def setup_lightningmodule(
    default_lightningmodule_kwargs, pond_datamodule, pond_dataset, trainer_kwargs
) -> SegmentationModel:
    lightningmodule_kwargs = copy.deepcopy(default_lightningmodule_kwargs)
    lightningmodule_kwargs["in_channels"] = pond_dataset.NUM_IN_CHANNELS
    lightningmodule_kwargs["num_classes"] = pond_dataset.NUM_CLASSES

    # wandb to reassign some LightningDataModule kwargs here
    lightningmodule_kwargs["learning_rate"] = wandb.config.learning_rate
    # lightningmodule_kwargs["lr_scheduler"] = wandb.config.lr_scheduler
    # lightningmodule_kwargs["segmentation_model"] = wandb.config.segmentation_model
    # lightningmodule_kwargs["encoder_name"] = wandb.config.encoder_name
    # lightningmodule_kwargs["encoder_weights"] = wandb.config.encoder_weights

    lightningmodule_kwargs = _setup_lr_scheduler(
        lightningmodule_kwargs, pond_datamodule, trainer_kwargs
    )

    model = SegmentationModel(**lightningmodule_kwargs)

    return model


def _setup_lr_scheduler(
    lightningmodule_kwargs, pond_datamodule, trainer_kwargs
) -> Dict[str, Any]:
    if lightningmodule_kwargs["lr_scheduler"] == "OneCycleLR":
        pond_datamodule.setup("fit")
        steps_per_epoch = len(pond_datamodule.train_dataloader())
        epochs = trainer_kwargs.get("max_epochs", None)
        train_time = trainer_kwargs.get("train_time", None)
        assert epochs is not None
        assert train_time is None
        lr_scheduler_config = {"steps_per_epoch": steps_per_epoch, "epochs": epochs}
        lightningmodule_kwargs["lr_scheduler_config"] = lr_scheduler_config

    return lightningmodule_kwargs


def setup_trainer(default_trainer_kwargs, batch_size) -> Trainer:
    trainer_kwargs = copy.deepcopy(default_trainer_kwargs)
    # set up logger
    csv_logger = CSVLogger(save_dir=MODELS_PATH, name=WANDB_PROJ_NAME)
    version = f"version_{csv_logger.version}"
    wandb_logger = WandbLogger(
        save_dir=MODELS_PATH, project=WANDB_PROJ_NAME, version=version
    )
    csv_logger = CSVLogger(save_dir=MODELS_PATH, name=WANDB_PROJ_NAME, version=version)
    experiment_config = {
        "max_epochs": trainer_kwargs["max_epochs"],
        "max_time": trainer_kwargs["max_time"],
        "precision": trainer_kwargs["precision"],
        "batch_size": batch_size,
    }
    wandb_logger.experiment.config.update(experiment_config)
    loggers = [csv_logger, wandb_logger]
    trainer_kwargs["logger"] = loggers
    trainer = Trainer(**trainer_kwargs)

    return trainer


def train_model() -> None:
    wandb.init(project=WANDB_PROJ_NAME)

    pond_dataset = PondDataset(**DEFAULT_DATASET_KWARGS)
    # batch_size = wandb.config.batch_size
    batch_size = DEFAULT_DATAMODULE_KWARGS["batch_size"]

    pond_datamodule = setup_datamodule(DEFAULT_DATAMODULE_KWARGS, batch_size)
    model = setup_lightningmodule(
        DEFAULT_LIGHTNINGMODULE_KWARGS,
        pond_datamodule,
        pond_dataset,
        DEFAULT_TRAINER_KWARGS,
    )
    trainer = setup_trainer(DEFAULT_TRAINER_KWARGS, batch_size)
    trainer.fit(model=model, datamodule=pond_datamodule)


def setup_sweep_configuration() -> Dict[str, Any]:
    monitor_metric = kwargs_dict["lightningmodule_kwargs"]["checkpoint_monitor_metric"]
    monitor_mode = kwargs_dict["lightningmodule_kwargs"]["checkpoint_monitor_mode"]
    if monitor_mode == "min":
        goal = "minimize"
    if monitor_mode == "max":
        goal = "maximize"

    sweep_configuration = {
        # use bayes if tuning learning rate only
        "method": "bayes",
        # "method": "random",
        "name": WANDB_PROJ_NAME,
        "metric": {"goal": goal, "name": monitor_metric},
        "parameters": {
            # "batch_size": {"values": [32, 64, 128]},
            "learning_rate": {
                "min": 5e-5,
                "max": 1e-3,
                "distribution": "log_uniform_values",
            },
            # "learning_rate": {"values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]},
            # "lr_scheduler": {"values": [None, "OneCycleLR"]},
            # "segmentation_model": {"values": ["Unet", "DeepLabV3Plus", "UnetPlusPlus"]},
            # "encoder_name": {
            #     "values": ["resnet34", "resnet50", "resnet101", "efficientnet-b3"]
            # },
            # "encoder_weights": {},
        },
    }

    return sweep_configuration


if __name__ == "__main__":
    sweep_configuration = setup_sweep_configuration()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=WANDB_PROJ_NAME)
    logger.info(f"Using sweep_id {sweep_id}")

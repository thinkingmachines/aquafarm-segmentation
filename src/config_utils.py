from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from lightning import LightningDataModule
from lightning.pytorch.callbacks import BackboneFinetuning, Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from segmentation_models_pytorch.encoders import get_preprocessing_params

from src.callbacks import WandbConfusionMatrix, WandbSegmentationMasks


def _build_dataset_kwargs(
    config_dict: Dict[str, Any],
    data_path: Path,
) -> List[Dict[str, Any]]:
    # append the actual data path to all the data directories
    parent_data_dir = config_dict["parent_data_dir"]
    data_dirs = config_dict["data_dirs"]
    data_dirs = {
        name: data_path / f"{parent_data_dir}/{path}"
        for name, path in data_dirs.items()
    }

    _params = get_preprocessing_params(
        encoder_name=config_dict["model_settings"]["encoder_name"],
        pretrained=config_dict["model_settings"]["encoder_weights"],
    )
    assert _params["input_space"] == "RGB"
    assert _params["input_range"] == [0, 1]
    # statistics of the images the pretrained neural net was trained on
    nn_normalizing_params = {"means": _params["mean"], "stdevs": _params["std"]}

    data_transform_settings = config_dict["data_transform_settings"]
    rescale_image_method = data_transform_settings["rescale_image_method"]

    # data augmentations if any
    augmentors = data_transform_settings.get("data_augmentation_transforms", None)

    # cloud cover related corrections if any
    cloud_cover_lookup_fpath = None
    cloud_cover_fname = config_dict.get("cloud_cover_lookup_fname", None)
    if cloud_cover_fname is not None:
        cloud_cover_lookup_fpath = data_path / parent_data_dir / cloud_cover_fname

    dataset_kwargs = {
        "imgs_root": data_dirs["train_imgs_root"],
        "masks_root": data_dirs["train_masks_root"],
        "nn_normalizing_params": nn_normalizing_params,
        "rescale_image_method": rescale_image_method,
        "augmentors": augmentors,
        "cloud_cover_lookup_fpath": cloud_cover_lookup_fpath,
    }

    datamodule_kwargs = {
        "train_imgs_root": data_dirs["train_imgs_root"],
        "train_masks_root": data_dirs["train_masks_root"],
        "predict_imgs_root": data_dirs["predict_imgs_root"],
        "predict_masks_root": data_dirs["predict_masks_root"],
        "batch_size": config_dict["dataloader_settings"]["batch_size"],
        "num_workers": config_dict["dataloader_settings"]["num_workers"],
        "nn_normalizing_params": nn_normalizing_params,
        "rescale_image_method": rescale_image_method,
        "augmentors": augmentors,
        "cloud_cover_lookup_fpath": cloud_cover_lookup_fpath,
    }

    return dataset_kwargs, datamodule_kwargs


def _build_trainer_kwargs(
    config_dict: Dict[str, Any],
    models_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    minutes = config_dict["trainer_settings"]["train_time_minutes"]
    if minutes is not None:
        train_time = timedelta(minutes=minutes)
    else:
        train_time = None

    callbacks = _setup_callbacks(config_dict)
    loggers = _setup_loggers(config_dict, models_path)

    trainer_kwargs = {
        "num_epochs": config_dict["trainer_settings"]["num_epochs"],
        "train_time": train_time,
        "callbacks": callbacks,
        "precision": config_dict["trainer_settings"]["precision"],
        "logger": loggers,
    }
    return trainer_kwargs


def _setup_callbacks(config_dict: Dict[str, Any]) -> List[Callback]:
    callback_names = config_dict["trainer_settings"]["callbacks"]
    if not callback_names:
        return None

    monitor_metric = config_dict["model_settings"]["checkpoint_monitor_metric"]
    filename = "{epoch}-{val_loss:.2f}-{" + monitor_metric + ":.2f}"
    model_checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        mode=config_dict["model_settings"]["checkpoint_monitor_mode"],
        filename=filename,
        auto_insert_metric_name=True,
    )
    callbacks = {
        "ModelCheckpoint": model_checkpoint,
        "WandbConfusionMatrix": WandbConfusionMatrix(),
        "WandbSegmentationMasks": WandbSegmentationMasks(),
        "BackboneFinetuning": BackboneFinetuning(),
    }
    wandb_callbacks = ["WandbConfusionMatrix", "WandbSegmentationMasks"]

    # if we're not using a WandbLogger, remove the Wandb callbacks
    logger_names = config_dict["trainer_settings"]["loggers"]
    if "WandbLogger" not in logger_names:
        for wandb_callback in wandb_callbacks:
            if wandb_callback in callback_names:
                raise ValueError(
                    f"Can't use {wandb_callback} because Wandb isn't set as a logger. Revise the config"
                )
        callback_names = [
            name for name in callback_names if name not in wandb_callbacks
        ]

    # select only the callbacks specified in the config
    callbacks = [callbacks[name] for name in callback_names]

    return callbacks


def _setup_loggers(
    config_dict: Dict[str, any], models_path: Optional[Path]
) -> Union[List[Logger], bool]:
    name = config_dict["parent_data_dir"]
    logger_names = config_dict["trainer_settings"]["loggers"]
    if models_path is None or not logger_names:
        return False

    valid_logger_names = ["CSVLogger", "WandbLogger"]
    for logger_name in logger_names:
        if logger_name not in valid_logger_names:
            raise ValueError(
                f"{logger_name} not a valid logger. Valid loggers are {valid_logger_names}"
            )
    use_csv_logger = "CSVLogger" in logger_names
    use_wandb_logger = "WandbLogger" in logger_names

    csv_logger = CSVLogger(save_dir=models_path, name=name)
    if use_wandb_logger:
        # hacky way for wandb to save to an incrementally versioned folder
        version = f"version_{csv_logger.version}"
        wandb_logger = WandbLogger(save_dir=models_path, project=name, version=version)
        csv_logger = CSVLogger(save_dir=models_path, name=name, version=version)

    minutes = config_dict["trainer_settings"]["train_time_minutes"]
    if minutes is not None:
        train_time = timedelta(minutes=minutes)
    else:
        train_time = None

    # data augmentations if any
    data_transform_settings = config_dict["data_transform_settings"]
    augmentors = data_transform_settings.get("data_augmentation_transforms", None)

    experiment_config = {
        "max_epochs": config_dict["trainer_settings"]["num_epochs"],
        "max_time": train_time,
        "batch_size": config_dict["dataloader_settings"]["batch_size"],
        "callbacks": config_dict["trainer_settings"]["callbacks"],
        "augmentors": augmentors,
        "precision": config_dict["trainer_settings"]["precision"],
    }

    loggers = []
    if use_csv_logger:
        csv_logger.log_hyperparams({"other_non_model_settings": experiment_config})
        loggers.append(csv_logger)

    if use_wandb_logger:
        wandb_logger.experiment.config.update(experiment_config)
        loggers.append(wandb_logger)

    return loggers


def _set_training_settings_defaults(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    if "callbacks" not in config_dict["trainer_settings"]:
        config_dict["trainer_settings"]["callbacks"] = False
    if "loggers" not in config_dict["trainer_settings"]:
        config_dict["trainer_settings"]["loggers"] = []
    if "precision" not in config_dict["trainer_settings"]:
        config_dict["trainer_settings"]["precision"] = "32-true"

    return config_dict


def build_kwargs_from_config(
    data_path: Path,
    config_fpath: Path,
    models_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function for generating kwargs for the Dataset, DataModule and LightningModule objects
    """

    with open(config_fpath, "r") as stream:
        config_dict = yaml.safe_load(stream)

    dataset_kwargs, datamodule_kwargs = _build_dataset_kwargs(config_dict, data_path)
    lightningmodule_kwargs = config_dict["model_settings"]

    if "trainer_settings" in config_dict.keys():
        config_dict = _set_training_settings_defaults(config_dict)
        trainer_kwargs = _build_trainer_kwargs(config_dict, models_path)
    else:
        trainer_kwargs = None

    misc_kwargs = {
        "random_seed": config_dict["random_seed"],
        "parent_dir": config_dict["parent_data_dir"],
    }

    kwargs_dict = {
        "dataset_kwargs": dataset_kwargs,
        "datamodule_kwargs": datamodule_kwargs,
        "lightningmodule_kwargs": lightningmodule_kwargs,
        "trainer_kwargs": trainer_kwargs,
        "misc_kwargs": misc_kwargs,
    }

    return kwargs_dict


def add_lr_scheduler_config(
    lightningmodule_kwargs: Dict[str, Any],
    trainer_kwargs: Dict[str, Any],
    datamodule: LightningDataModule,
) -> Dict[str, Any]:
    """
    Adding lr scheduler config to the lightningmodule for the OneCycleLR
    """

    if lightningmodule_kwargs["lr_scheduler"] == "OneCycleLR":
        datamodule.setup("fit")
        steps_per_epoch = len(datamodule.train_dataloader())
        epochs = trainer_kwargs.get("max_epochs", None)
        train_time = trainer_kwargs.get("train_time", None)
        assert epochs is not None
        assert train_time is None
        lr_scheduler_config = {"steps_per_epoch": steps_per_epoch, "epochs": epochs}
        lightningmodule_kwargs["lr_scheduler_config"] = lr_scheduler_config

    return lightningmodule_kwargs


def __lookup_fix_other_datasets(
    data_path: Path,
    config_dict: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    datamodule_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    parent_data_dir = config_dict["parent_data_dir"]

    # lookup file specific to crop delination
    data_split_fname = config_dict.get("data_split_fname", None)
    if data_split_fname is not None:
        data_split_fpath = data_path / parent_data_dir / data_split_fname
        dataset_kwargs["data_split_fpath"] = data_split_fpath
        datamodule_kwargs["data_split_fpath"] = data_split_fpath

    # lookup file specific to oxford pets
    fnames_paths = config_dict.get("fnames_paths", None)
    if fnames_paths is not None:
        train_fnames_fpath = (
            data_path / parent_data_dir / fnames_paths["train_fnames_fpath"]
        )
        predict_fnames_fpath = (
            data_path / parent_data_dir / fnames_paths["predict_fnames_fpath"]
        )

        dataset_kwargs["fnames_fpath"] = train_fnames_fpath
        datamodule_kwargs["train_fnames_fpath"] = train_fnames_fpath
        datamodule_kwargs["predict_fnames_fpath"] = predict_fnames_fpath

    return dataset_kwargs, datamodule_kwargs

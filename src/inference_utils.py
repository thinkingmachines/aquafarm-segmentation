import os
from pathlib import Path
from typing import Optional, Union

from loguru import logger


def get_latest_log_version(model_artifacts_dir: Union[str, Path]) -> int:
    model_artifacts_dir = Path(model_artifacts_dir)
    version_dirs = os.listdir(model_artifacts_dir)
    version_indicies = [
        int(x.split("_")[1]) for x in version_dirs if x.startswith("version_")
    ]
    latest_ver = max(version_indicies)

    return latest_ver


def get_checkpoint_fpath(
    model_artifacts_dir: Union[str, Path], version_num: Optional[int] = None
) -> Path:
    if version_num is None:
        version_num = get_latest_log_version(model_artifacts_dir)
        logger.info(
            f"Since version is unspecified, getting latest version {version_num}"
        )
    version_dir = model_artifacts_dir / f"version_{version_num}/"
    checkpoint_dir = version_dir / "checkpoints"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No checkpoints folder found in {version_dir}. The checkpoint might have not yet been saved"
        )
    checkpoint_file = os.listdir(checkpoint_dir)

    if len(checkpoint_file) > 1:
        raise ValueError(
            f"Expected to have 1 checkpoint in path but instead had these: {checkpoint_file}"
        )

    checkpoint_fpath = checkpoint_dir / checkpoint_file[0]
    return checkpoint_fpath


def get_checkpoint_hparams(
    model_artifacts_dir: Union[str, Path], version_num: Optional[int] = None
) -> Path:
    if version_num is None:
        version_num = get_latest_log_version(model_artifacts_dir)
        logger.info(
            f"Since version is unspecified, getting latest version {version_num}"
        )

    hparams_path = model_artifacts_dir / f"version_{version_num}/hparams.yaml"

    return hparams_path

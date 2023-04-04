from typing import Union

import segmentation_models_pytorch as smp
import torch
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import (
    FCN,
    DeepLabV3,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
)

TORCHVISION_ARCH_NAMES = ["fcn_torchvision", "deeplabv3_torchvision"]
TORCHVISION_ENCODER_NAMES = ["resnet50", "resnet101"]
TORCHVISION_ENCODER_WEIGHTS = ["imagenet_v1", "imagenet_v2"]
# use this as default weights for torchvision
DEFAULT_TORCHVISION_ENCODER_WEIGHTS = "imagenet_v2"


def create_model(
    arch: str,
    encoder_name: str,
    encoder_weights: str,
    in_channels: int,
    num_classes: int,
    return_backbone: bool = True,
) -> torch.nn.Module:
    """
    Create a semantic segmentation model either from
    segmentation models pytorch or from torchvision
    """
    is_torchvision = arch in TORCHVISION_ARCH_NAMES

    if not is_torchvision:
        model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        backbone = model.encoder
    else:
        model = create_torchvision_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        backbone = model.backbone

    output = model
    if return_backbone:
        output = (model, backbone)

    return output


def create_torchvision_model(
    arch: str,
    encoder_name: str,
    encoder_weights: str,
    in_channels: int,
    num_classes: int,
):
    if in_channels != 3:
        raise ValueError(f"in_channels should be 3, but instead got {in_channels}")

    if encoder_weights == "imagenet":
        encoder_weights = DEFAULT_TORCHVISION_ENCODER_WEIGHTS
    else:
        raise ValueError(f"encoder_weights {encoder_weights} is not supported")

    verify_torchvision_params(arch, encoder_name, encoder_weights)

    weights_backbone = get_torchvision_weights_backbone(encoder_name, encoder_weights)

    model = get_torchvision_architecture(
        arch, encoder_name, weights_backbone, num_classes
    )

    return model


def verify_torchvision_params(
    arch: str, encoder_name: str, encoder_weights: str
) -> None:
    if arch not in TORCHVISION_ARCH_NAMES:
        raise ValueError(
            f"{arch} not a valid arch. The valid values are {TORCHVISION_ARCH_NAMES}"
        )

    if encoder_name not in TORCHVISION_ENCODER_NAMES:
        raise ValueError(
            f"{encoder_name} not a valid encoder_name. The valid values are {TORCHVISION_ENCODER_NAMES}"
        )

    if encoder_weights not in TORCHVISION_ENCODER_WEIGHTS:
        raise ValueError(
            f"{encoder_weights} not a valid encoder_weights. The valid values are {TORCHVISION_ENCODER_WEIGHTS}"
        )


def get_torchvision_weights_backbone(
    encoder_name: str, encoder_weights: str
) -> Union[ResNet50_Weights, ResNet101_Weights]:
    if encoder_name == "resnet50" and encoder_weights == "imagenet_v1":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V1
    elif encoder_name == "resnet50" and encoder_weights == "imagenet_v2":
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2
    elif encoder_name == "resnet101" and encoder_weights == "imagenet_v1":
        weights_backbone = ResNet101_Weights.IMAGENET1K_V1
    elif encoder_name == "resnet101" and encoder_weights == "imagenet_v2":
        weights_backbone = ResNet101_Weights.IMAGENET1K_V2
    else:
        raise ValueError(
            f"Uncrecognized encoder_name and encoder_weights pair {encoder_name}, {encoder_weights}"
        )

    return weights_backbone


def get_torchvision_architecture(
    arch: str,
    encoder_name: str,
    weights_backbone: Union[ResNet50_Weights, ResNet101_Weights],
    num_classes: int,
) -> Union[DeepLabV3, FCN]:
    if arch == "fcn_torchvision" and encoder_name == "resnet50":
        model = fcn_resnet50(num_classes=num_classes, weights_backbone=weights_backbone)
    elif arch == "fcn_torchvision" and encoder_name == "resnet101":
        model = fcn_resnet101(
            num_classes=num_classes, weights_backbone=weights_backbone
        )
    elif arch == "deeplabv3_torchvision" and encoder_name == "resnet50":
        model = deeplabv3_resnet50(
            num_classes=num_classes, weights_backbone=weights_backbone
        )
    elif arch == "deeplabv3_torchvision" and encoder_name == "resnet101":
        model = deeplabv3_resnet101(
            num_classes=num_classes, weights_backbone=weights_backbone
        )
    else:
        raise ValueError(
            f"Unrecognized arch and encoder_name pair {arch}, {encoder_name}"
        )

    return model

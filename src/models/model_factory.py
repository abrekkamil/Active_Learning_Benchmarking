"""
Model factory for Active Learning Benchmark.

Creates models based on config.model_name.
All returned models follow the unified API:

- train_epoch(dataset, epoch, total_epochs)
- evaluate(dataset)
- predict(images)
- get_uncertainty(images)
"""

from typing import Any

from .unet_model import UNetModel
from .deeplab_model import DeepLabV3Model
from .segformer_model import SegFormerModel
from .maskrcnn_model import MaskRCNNModel
from .yolo_model import YOLOv8Model
from .multilabel_model import MultiLabelClassificationModel

def build_model(model_name: str, num_classes: int, device, config) -> Any:
    """
    Build model from name.

    Parameters
    ----------
    model_name : str
        Name of model to create.

    num_classes : int
        Number of dataset classes.

    device : torch.device
        Device for model.

    config : config object
        Experiment configuration.

    Returns
    -------
    model
        Model instance implementing unified API.
    """

    model_name = model_name.lower()

    # ------------------------
    # Segmentation models
    # ------------------------

    if model_name == "unet":
        return UNetModel(num_classes, device, config)

    elif model_name == "deeplabv3":
        return DeepLabV3Model(num_classes, device, config)

    elif model_name == "segformer":
        return SegFormerModel(num_classes, device, config)

    # ------------------------
    # Detection / instance segmentation
    # ------------------------

    elif model_name == "maskrcnn":
        return MaskRCNNModel(num_classes, device, config)

    elif model_name == "yolo":
        task = getattr(config, "yolo_task", "detect")

        return YOLOv8Model(
            num_classes=num_classes,
            device=device,
            config=config,
            task=task,
        )
    elif model_name in ("resnet50_multilabel", "multilabel"):
        return MultiLabelClassificationModel(num_classes, device, config)
    # ------------------------
    # Error
    # ------------------------

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Available models: unet, deeplabv3, segformer, maskrcnn, yolo"
        )
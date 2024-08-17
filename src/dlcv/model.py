import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.models.detection import FasterRCNN, FCOS, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import FCOS, RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import mobilenet_v2
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def get_fcos_model(num_classes, backbone_name="resnet50"):
    # Load a pre-trained ResNet-50 backbone with FPN
    backbone = resnet_fpn_backbone(backbone_name, weights='IMAGENET1K_V1', trainable_layers=5)

    # Initialize the FCOS model with the modified backbone
    model = FCOS(backbone, num_classes=num_classes)

    # Set the model in training mode
    model.train()

    return model


def get_fasterrcnn_model(num_classes, backbone_name="resnet50"):
    # Load a pre-trained ResNet-50 backbone with FPN
    backbone = resnet_fpn_backbone(backbone_name, weights='IMAGENET1K_V1', trainable_layers=5)

    # Initialize the FCOS model with the modified backbone
    model = FasterRCNN(backbone, num_classes=num_classes)

    # Set the model in training mode
    model.train()

    return model

def get_retinanet_model(num_classes, backbone_name="resnet50"):
    # Load a pre-trained ResNet-50 backbone with FPN
    backbone = resnet_fpn_backbone(backbone_name, weights='IMAGENET1K_V1', trainable_layers=5)

    # Initialize the FCOS model with the modified backbone
    model = RetinaNet(backbone, num_classes=num_classes)

    # Set the model in training mode
    model.train()

    return model

def get_model(num_classes, model_type, backbone_name):
    if model_type == "fcos":
        return get_fcos_model(num_classes, backbone_name)
    elif model_type == "fasterrcnn":
        return get_fasterrcnn_model(num_classes, backbone_name)
    elif model_type == "retinanet":
        return get_retinanet_model(num_classes, backbone_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

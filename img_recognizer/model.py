"""
This module creates and initializes the model.
"""
import torch
import torch.nn as nn


def resnet18(freeze_parameters: bool = True) -> nn.Module:
    """
    Load a pre-trained ResNet-18 model with the option to freeze its parameters and modify the output layer.

    Args:
        freeze_parameters (bool, optional): If True, freeze the parameters of the loaded ResNet-18 model.
            Defaults to True.

    Returns:
        model (nn.Module): The ResNet-18 model with modified output layer.
    """
    model = torch.hub.load("pytorch/vision:v0.10.0",
                           "resnet18", pretrained=True)

    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    return model

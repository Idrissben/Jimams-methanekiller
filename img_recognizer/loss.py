"""
This module provides a function calculate the loss of the model.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader


def calculate_loss(
    model: nn.Module, dataloader: DataLoader, batch_size: int = 16
) -> float:
    """
    Calculate the total loss of a model on a dataset.

    Args:
        model (nn.Module): The PyTorch model for which the loss will be calculated.
        dataloader (DataLoader): DataLoader providing batches of data and labels.
        batch_size (int, optional): Batch size for processing data. Defaults to 16.

    Returns:
        loss_result (float): The total loss for the provided dataset.
    """
    loss_fn = nn.CrossEntropyLoss()
    loss_epoch_test = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            target = torch.zeros((min(len(labels), batch_size)))

            for i in range(min(len(labels), batch_size)):
                target[i] = labels[i]

            target = target.type(torch.LongTensor)
            inputs = inputs.type(torch.FloatTensor)

            outputs = model(inputs)
            loss = loss_fn(outputs, target)
            loss_epoch_test += loss
    loss_result = loss_epoch_test.item()

    return loss_result

"""
This module provides functions to train the model.
"""
from typing import Callable
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from loss import calculate_loss
from evaluation import success_rate
import matplotlib.pyplot as plt


def train(
    model: torch.nn.Module,
    data_train: Dataset,
    data_validation: Dataset,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lr: float,
    epochs: int,
    batch_size: int,
    print_progress: bool,
) -> torch.nn.Module:
    """
    Train a neural network model using the specified data and hyperparameters.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        data_train (Dataset): Training dataset.
        data_validation (Dataset): Validation dataset.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function.
        lr (float): The learning rate.
        epochs (int): The number of training epochs.
        batch_size (int): Batch size for training.
        print_progress (bool): Whether or not to print losses and plots.

    Returns:
        model (torch.nn.Module): The trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(
        data_validation, batch_size=batch_size, shuffle=True
    )
    training_loss = []
    validation_loss = []
    success_rate_ = [0]
    best_val_loss = float("inf")
    early_stopping = 0

    for _ in tqdm(range(epochs), desc="Total Progress: "):
        loss_epoch_train = 0

        for inputs, labels in dataloader_train:
            target = torch.zeros((min(len(labels), batch_size)))

            for i in range(min(len(labels), batch_size)):
                target[i] = labels[i]

            target = target.type(torch.LongTensor)
            inputs = inputs.type(torch.FloatTensor)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, target)
            loss_epoch_train += loss
            loss.backward()
            optimizer.step()

        if print_progress:
            print(f"Training Loss: {loss_epoch_train.detach()}")
            training_loss.append(loss_epoch_train.detach())

            val_loss = calculate_loss(model, dataloader_validation, batch_size)
            print(f"Validation Loss: {val_loss}")
            validation_loss.append(val_loss)

            accuracy_rate = success_rate(
                model, dataloader_validation, batch_size)
            print("Model accuracy: {accuracy_rate} %")
            success_rate_.append(accuracy_rate)

            print("Epoch done.")
            print("-------------------------------------")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
        else:
            early_stopping += 1
            if early_stopping >= 2:
                print("Early stopping")
                break

    if print_progress:
        plt.plot(training_loss)
        plt.xlabel("Epoch number")
        plt.ylabel("Training Loss")
        plt.title("Evolution of training loss with the number of epochs")
        plt.show()

        plt.plot(validation_loss)
        plt.xlabel("Epoch number")
        plt.ylabel("Validation Loss")
        plt.title("Evolution of validation loss with the number of epochs")
        plt.show()

        plt.plot(success_rate_)
        plt.xlabel("Epoch number")
        plt.ylabel("Model Accuracy (%)")
        plt.title("Evolution of our model's accuracy with the number of epochs")
        plt.show()

    return model

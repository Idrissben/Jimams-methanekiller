"""
This module provides functions to evaluate the accuracy of the model.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score


def success_rate(model: torch.nn.Module, data: DataLoader, batch_size: int) -> float:
    """
    Calculate the success rate (accuracy) of a model on a dataset.

    Args:
        model (torch.nn.Module): The PyTorch model for which the success rate will be calculated.
        data (DataLoader): DataLoader providing batches of data and labels.
        batch_size (int): Batch size for processing data.

    Returns:
        rate (float): The success rate (accuracy) of the model on the dataset.
    """
    N = len(data)
    counter = 0

    with torch.no_grad():
        for inputs, labels in data:
            outputs = model(inputs)
            target = torch.zeros((min(len(labels), batch_size)))
            for i in range(min(len(labels), batch_size)):
                target[i] = labels[i]
            i = 0
            for output in outputs:
                if torch.argmax(output) == target[i]:
                    counter += 1
                i += 1
            l = len(target)
    rate = (counter / ((N - 1) * batch_size + l)) * 100
    return rate


def conf_mat(model: torch.nn.Module, test_data: DataLoader, batch_size: int) -> None:
    """
    Compute and display a confusion matrix for a model's predictions on a test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model for which the confusion matrix will be calculated.
        test_data (DataLoader): DataLoader providing batches of test data and labels.
        batch_size (int): Batch size for processing data.

    Returns:
        None
    """
    nb_classes = 2
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    dataloader_test = DataLoader(
        test_data, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_test, desc="Total Progress: "):
            inputs = inputs.type(torch.FloatTensor)
            outputs = model(inputs)

            for i in range(min(len(outputs), batch_size)):
                pred = torch.argmax(outputs[i]).item()
                confusion_matrix[labels[i]][pred] += 1

    plt.figure(figsize=(7, 5))
    class_names = ["Contains Plume", "Does Not Contain Plume"]

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names
    ).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=8
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=8
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_auc(model: torch.nn.Module, data: DataLoader, batch_size: int) -> float:
    """
    Calculate the Area Under the Receiver Operating Characteristic (ROC) Curve (AUC) for a model's predictions on a dataset.

    Args:
        model (torch.nn.Module): The PyTorch model for which the AUC will be calculated.
        data (DataLoader): DataLoader providing batches of data and labels.
        batch_size (int): Batch size for processing data.

    Returns:
        auc (float): The calculated AUC score.
    """
    model.eval()
    all_labels = []
    all_scores = []
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Calculating AUC: "):
            inputs = inputs.type(torch.FloatTensor)
            outputs = model(inputs)
            scores = torch.sigmoid(outputs)[:, 1]

            all_labels.extend(labels.numpy())
            all_scores.extend(scores.numpy())

    auc = roc_auc_score(all_labels, all_scores)
    return auc

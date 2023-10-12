import torch
import torchvision.transforms as transforms
from PIL import Image
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
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'resnet18', pretrained=True)

    if freeze_parameters:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    return model


class Classifier:
    def __init__(self, model_path):
        # Create a new ResNet-50 model with random weights
        self.model = resnet18()

        self.model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode
        self.model.eval()
        # Define the image transformations to be applied
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_path):
        # Load the image from the specified file
        image = Image.open(image_path)
        # Apply the specified transformations to the image
        image = self.transform(image)
        # Add a batch dimension to the image
        image = image.unsqueeze(0)
        # Use the model to predict the class probabilities for the image
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # Return the predicted class and its probability
        return torch.argmax(probabilities).item(), probabilities[torch.argmax(probabilities)].item()

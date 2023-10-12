from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

def convert_to_rgb(image_path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to the RGB mode if it's not already in that format.

    Args:
        image_path (str): The path to the image file.

    Returns:
        image (Image.Image): The loaded image in RGB mode.
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

class TrainData(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        base_path: str,
        transform: transforms.Compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        augmentation: transforms.Compose = None
    ):
        """
        Custom dataset class for training data.

        Args:
            metadata (pd.DataFrame): A DataFrame containing information about the dataset, including image paths and labels.
            base_path (str): The base path for image files.
            transform (transforms.Compose, optional): A composition of image transformations. Default is to resize, convert to tensor, and normalize.
            augmentation (transforms.Compose, optional): A composition of data augmentation transformations.

        """
        self.transform = transform
        self.augmentation = augmentation
        self.metadata = metadata
        self.base_path = base_path

    def __len__(self) -> int:
        return self.metadata.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        label = 1 if self.metadata['plume'].iloc[idx] == 'yes' else 0
        img_path = self.base_path + self.metadata['path'].iloc[idx] + '.tif'
        image = convert_to_rgb(img_path)

        if self.augmentation:
            image = self.augmentation(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    
class TestData(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        base_path: str,
        transform: transforms.Compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    ):
        """
        Custom dataset class for testing data.

        Args:
            metadata (pd.DataFrame): A DataFrame containing information about the dataset, including image paths.
            base_path (str): The base path for image files.
            transform (transforms.Compose, optional): A composition of image transformations. Default is to resize, convert to tensor, and normalize.
        """
        self.transform = transform
        self.metadata = metadata
        self.base_path = base_path

    def __len__(self) -> int:
        return self.metadata.shape[0]

    def __getitem__(self, idx: int) -> Image.Image:
        img_path = self.base_path + 'images/' + str(self.metadata['date'].iloc[idx]) + '_methane_mixing_ratio_' + self.metadata['id_coord'].iloc[idx] + '.tif'
        image = convert_to_rgb(img_path)

        if self.transform:
            image = self.transform(image)

        return image
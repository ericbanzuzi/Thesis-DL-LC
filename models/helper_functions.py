# Write a custom dataset class (inherits from torch.utils.data.Dataset)
import os
import pathlib
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
from typing import Tuple, Dict, List
import random
import numpy as np


# based on: https://www.learnpytorch.io/04_pytorch_custom_datasets/#
# Make function to find classes in target directory
def find_classes(directory: str):
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.
    Returns:
        tuple: (list_of_class_names, dict(class_name: idx...))
    Example:
        find_classes("datasets/train")
         (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# based on: https://www.learnpytorch.io/04_pytorch_custom_datasets/#
class VideoFolderCustom(Dataset):
    # Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None, permute=False) -> None:
        # 3. Create class attributes
        # Get all video paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.mp4"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)
        # Permutation needed?
        self.permute = permute

    def load_video(self, index: int) -> torch.Tensor:
        """Opens an image via a path and returns it."""
        video_path = self.paths[index]
        frames, _, _ = read_video(str(video_path), output_format="TCHW")
        frames = frames.div(255.0)
        return frames

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.paths)

    # Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """ Returns one sample of data, data and label (X, y). """
        vid = self.load_video(index)
        class_name = self.paths[index].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        if self.permute:
            vid = vid.permute((1, 0, 2, 3))  # matches the pytorch inbuilt models

        # Transform if necessary
        if self.transform:
            return self.transform(vid), class_idx  # return data, label (X, y)
        else:
            return vid, class_idx  # return data, label (X, y)


# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        print(targ_image.size())
        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image[0].permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()


train_dir = '../datasets/train/Recognition/ROI 2'
train_data_custom = VideoFolderCustom(targ_dir=train_dir)

print(train_data_custom)

print(len(train_data_custom))
print(train_data_custom.classes)
print(train_data_custom.class_to_idx)

# Display random images from ImageFolderCustom Dataset
display_random_images(train_data_custom,
                      n=12,
                      classes=train_data_custom.classes,
                      seed=None)  # Try setting the seed for reproducible images

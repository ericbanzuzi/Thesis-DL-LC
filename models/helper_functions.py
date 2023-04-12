# Write a custom dataset class (inherits from torch.utils.data.Dataset)
import os
import pathlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_video
from typing import Tuple, List
import random
from tqdm import tqdm


# based on: https://www.learnpytorch.io/04_pytorch_custom_datasets/#
# Make function to find classes in target directory
def find_classes(directory: str):
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Example: find_classes("datasets/train") returns (["class_1", "class_2"], {"class_1": 0, ...})

    Args:
        directory (str): target directory to load classnames from.
    Returns:
        tuple: (list_of_class_names, dict(class_name: idx...))
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
    def __init__(self, targ_dir: str, transform=None, permute=True, augmented=False, ROI=2) -> None:
        # 3. Create class attributes
        # Get all video paths
        if augmented:
            self.paths = list(pathlib.Path(targ_dir).glob(f"*/*x{ROI}.mp4"))
        else:
            self.paths = list(pathlib.Path(targ_dir).glob("*/*.mp4"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)
        print(self.class_to_idx)
        # Permutation needed?
        self.permute = permute

    def load_video(self, index: int) -> torch.Tensor:
        """Opens an image via a path and returns it."""
        video_path = self.paths[index]
        frames, _, _ = read_video(str(video_path), output_format='TCHW', pts_unit='sec')
        frames = frames.div(255.0)
        return frames[-32:]

    # Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
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


# Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    """
    Displays a random set of first frames from the videos of a dataset

    :param dataset: a torch dataset
    :param classes: a list of class names in the dataset
    :param n: number of images
    """
    # Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # Set random seed
    if seed:
        random.seed(seed)

    # Get random sample indices
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Setup plot
    plt.figure(figsize=(16, 8))

    #  Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
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


def step_test(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    """
    Performs a step of testing the model on a validation set during training

    :param model: the model to be trained
    :param dataloader: the test dataset loader
    :param loss_fn: the loss function used for training
    :param device: the device used for training (e.g. GPU)
    :return: test_loss, test_acc
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# Take in various parameters required for training and test steps, save the best model
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device,
          result_path,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    """
    Trains a model using the usual Torch training loop

    :param model: model to be trained
    :param train_dataloader: the train dataloader
    :param test_dataloader: the test dataloader
    :param optimizer: optimizer used for the training
    :param device: device uused for the training
    :param result_path: the path where the results from the training are stored
    :param loss_fn: loss function for the training
    :param epochs: number of epochs used for the training
    """

    # Create empty results txt file
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    with open(result_path+'/training_results.txt', 'a+') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')
    f.close()

    best_test_acc = 0
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        loop = tqdm(train_dataloader)
        # Put model in train mode
        model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(loop):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=train_loss/(batch+1), acc=train_acc/(batch+1))

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        test_loss, test_acc = step_test(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results
        with open(result_path+'/training_results.txt', 'a+') as f:
            f.write(f'{epoch + 1},{train_loss},{train_acc},{test_loss},{test_acc}\n')
        f.close()

        # save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': test_acc,
            }, f'{result_path}/best_model.pth')
    return


def plot_training_path(data, model_title):
    """
    Visualizes the training process, more specifically, the loss and accuracy

    :param data: a dataframe containing the training history
    :param model_title: the title for the plot
    """
    # loss and accuracy plot
    fig, subplots = plt.subplots(1, 2, figsize=(14, 6))
    subplots[0].plot(data['epoch'], data['train_loss'], label='train')
    subplots[0].plot(data['epoch'], data['test_loss'], label='test')
    subplots[0].set_title(f'{model_title} Loss')
    subplots[0].set_ylabel('Loss')
    subplots[0].set_xlabel('Epoch')
    subplots[0].legend()

    subplots[1].plot(data['epoch'], data['train_acc'], label='train')
    subplots[1].plot(data['epoch'], data['test_acc'], label='test')
    subplots[1].set_title(f'{model_title} Accuracy')
    subplots[1].set_ylabel('Accuracy')
    subplots[1].set_xlabel('Epoch')
    subplots[1].legend()
    plt.show()
    return


def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device):
    """
    Performs the inferences from a model and a test dataset

    :param model: the model to be evaluated
    :param dataloader: the dataloader with the test data
    :param device: device used for the inferences
    :return: predictions, actual_labels
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    predictions, actual_labels = [], []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            predictions.extend(test_pred_labels.tolist())
            actual_labels.extend(y.tolist())

    return predictions, actual_labels

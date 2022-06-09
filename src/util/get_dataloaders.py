import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import math
import os
import config


DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_dataloaders(data_dir, batch_size):
    """
    Returns the (train-test-validation) dataloaders and a list containing the target classes.

    Parameters
    ----------
    data_dir: str
        Path at which the data is stored
    batch_size: int
        Batch size for the dataloaders

    Returns
    -------
    trainloader: PyTorch Dataloader
        Dataloader over the training dataset
    valloader: PyTorch Dataloader
        Dataloader over the validation dataset
    testloader: PyTorch Dataloader
        Dataloader over the test dataset
    classes: list
        List of all the target classes

    """
    dataset = datasets.ImageFolder(data_dir, transform=DATA_TRANSFORM)
    dataset_size = len(dataset)

    train_set_size = math.ceil(dataset_size * (1 - config.test_dataset_ratio))
    test_set_size = dataset_size - train_set_size
    val_set_size = math.ceil(dataset_size * (1 - config.test_dataset_ratio) * config.val_dataset_ratio)
    final_train_set_size = train_set_size - val_set_size

    train_set, test_set = random_split(dataset, [train_set_size, test_set_size])
    train_set, val_set = random_split(train_set, [final_train_set_size, val_set_size])

    torch.manual_seed(0)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    torch.manual_seed(torch.initial_seed())

    classes = os.listdir(data_dir)
    classes.sort()

    if classes[0] == '.DS_Store':
        del classes[0]

    return trainloader, valloader, testloader, classes

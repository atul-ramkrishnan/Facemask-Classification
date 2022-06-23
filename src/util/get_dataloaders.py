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

def get_data(data_dir, for_bias_test=False, for_kfold=False, batch_size=32):
    """

    Parameters
    ----------
    data_dir: str
        Path at which the data is stored
    for_bias_test: bool
        Whether the data is for testing the bias in the dataset
    for_kfold: bool
        Whether the data is for k-fold cross validation
    batch_size: int
        Batch size for the dataloaders

    Returns
    -------
    rainloader: PyTorch Dataloader
        Dataloader over the training dataset
    valloader: PyTorch Dataloader
        Dataloader over the validation dataset
    testloader: PyTorch Dataloader
        Dataloader over the test dataset
    classes: list
        List of all the target classes
    """

    classes = os.listdir(data_dir)
    classes.sort()

    if classes[0] == '.DS_Store':
        del classes[0]

    dataset = datasets.ImageFolder(data_dir, transform=DATA_TRANSFORM)
    dataset_size = len(dataset)

    if for_bias_test:
        testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return testloader, classes
    elif for_kfold:
        return dataset, classes
    else:
        train_set_size = math.ceil(dataset_size * (1 - config.test_dataset_ratio))
        test_set_size = dataset_size - train_set_size
        val_set_size = math.ceil(dataset_size * (1 - config.test_dataset_ratio) * config.val_dataset_ratio)
        final_train_set_size = train_set_size - val_set_size
        torch.manual_seed(0)
        train_set, test_set = random_split(dataset, [train_set_size, test_set_size])
        train_set, val_set = random_split(train_set, [final_train_set_size, val_set_size])
        torch.manual_seed(torch.initial_seed())

        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return trainloader, valloader, testloader, classes

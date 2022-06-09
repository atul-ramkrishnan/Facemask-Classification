import torch.nn as nn
from torch.optim import Adam
import sys
from models import models
from util.get_dataloaders import get_dataloaders
from train import train
from test import test
from demo import demo
import config


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Please provide arguments to run the file')

    operation = sys.argv[1]
    mtype = sys.argv[2]

    if mtype == "BaselineCNN":
        model = models.BaselineCNN()
    elif mtype == "CNNTwoLayerMaxPooling":
        model = models.CNNTwoLayerMaxPooling()
    elif mtype == "CNNThreeLayerMaxPooling":
        model = models.CNNThreeLayerMaxPooling()
    else:
        model = models.CNNThreeLayerMaxPooling()

    trainloader, valloader, testloader, classes = get_dataloaders(config.data_dir, config.batch_size)
    if operation == "train":
        # Select the loss criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train(model, config.epochs, criterion, optimizer, trainloader, valloader, config.device,
              config.saved_models_dir + '/' + mtype + '.pth')
    elif operation == "test":
        test(model, testloader, classes, config.device, config.saved_models_dir + '/' + mtype + '.pth')
    elif operation == "demo":
        demo(model, classes, config.device, config.saved_models_dir + '/' + mtype + '.pth', config.demo_dir)
    else:
        raise ValueError('Invalid operation.')

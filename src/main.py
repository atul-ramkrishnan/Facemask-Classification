import torch.nn as nn
from torch.optim import Adam
import sys
from models import models
from util.get_dataloaders import get_data
from train import train, train_kfold
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

    if operation == "train_kfold":
        dataset, classes = get_data(config.data_dir, False, True)
    elif operation == 'test_bias':
        testloader, classes = get_data(config.bias_test_dir, True, False, config.batch_size)
    else:
        trainloader, valloader, testloader, classes = get_data(config.data_dir, False, False, config.batch_size)

    if operation == "train":
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train(model, config.epochs, criterion, optimizer, trainloader, valloader, config.device,
              config.saved_models_dir + '/' + mtype + '.pth')

    elif operation == "train_kfold":
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_kfold(model, config.epochs, config.num_folds, criterion, optimizer, dataset, config.device,
                    classes, config.saved_models_dir + '/' + mtype)

    elif operation == "test":
        test(model, testloader, classes, config.device, config.saved_models_dir + '/' + mtype + '.pth')

    elif operation == "test_bias":
        test(model, testloader, classes, config.device, config.saved_models_dir + '/' + mtype + '.pth')

    elif operation == "demo":
        demo(model, classes, config.device, config.saved_models_dir + '/' + mtype + '.pth', config.demo_dir)

    else:
        raise ValueError('Invalid operation.')

import torch
from util.visualize import plot_training_graphs
from util.early_stopping import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from test import getAccuracy, showConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def train(model, epochs, criterion, optimizer, train_loader, val_loader,
                      device, model_save_path):
    """
    Trains the model and saves the one with the lowest validation loss.

    Parameters
    ----------
    model: PyTorch model
        PyTorch model that is to be trained
    epochs: int
        Maximum number of epochs that the model is to be trained for
    criterion: PyTorch criterion
        Criterion for calculating loss.
    optimizer: PyTorch optimizer
        Optimizer used for backpropagating gradients
    train_loader: PyTorch Dataloader
        PyTorch Dataloader over the training dataset
    val_loader: PyTorch Dataloader
        PyTorch Dataloaer over the validation dataset
    device: torch.device
        Device on which the model is to be trained (CPU/GPU)
    model_save_path: str
        The path to which the model is to be saved after training

    Returns
    -------
    None
    """
    print("STARTING TRAINING...")
    model.to(device)

    training_loss_list = []
    val_loss_list = []
    model_accuracy_list = []

    early_stopping = EarlyStopping(
        patience=3,
        save_path=model_save_path,
        min_delta=0.001)

    for epoch in range(1, epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0

        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss / len(train_loader)
        training_loss_list.append(train_loss_value)

        # Validation loop
        with torch.no_grad():
            model.eval()
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                running_val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == labels).sum().item()

        val_loss_value = running_val_loss / len(val_loader)
        accuracy = (100 * running_accuracy / total)
        val_loss_list.append(val_loss_value)
        model_accuracy_list.append(accuracy)

        print(
            f"Epoch {epoch} => Training loss: {train_loss_value:.4f}, Validation loss: {val_loss_value:.4f}, Accuracy: {accuracy:.4f}%")

        if early_stopping(model, val_loss_value):
            break

    plot_training_graphs(epoch, training_loss_list, val_loss_list, [a / 100 for a in model_accuracy_list])

# References
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
# https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
def train_kfold(model, epochs, num_folds, criterion, optimizer, dataset,
                      device, classes, model_save_path):
    # For fold results
    results = {}
    for target_class in classes:
        results[target_class] = {}
        results[target_class] = {}
        results[target_class] = {}
        results[target_class] = {}
    for target_class in classes:
        results[target_class]['accuracy'] = []
        results[target_class]['precision'] = []
        results[target_class]['recall'] = []
        results[target_class]['fscore'] = []

    results['aggregate_accuracy'] = []

    torch.manual_seed(0)
    kfold = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            sampler=test_subsampler)

        model.apply(reset_weights)

        for epoch in range(1, epochs + 1):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                train_loss = criterion(outputs, labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_save_path + '_fold_' + str(fold) + '.pth')

        # Evaluation for this fold
        correct_class_map = {label: 0 for label in classes}
        pred_class_map = {label: 0 for label in classes}
        correct = 0
        total = 0

        predlist = torch.zeros(0, dtype=torch.long, device='cpu')
        labellist = torch.zeros(0, dtype=torch.long, device='cpu')

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, predictions = torch.max(model(images), 1)

                predlist = torch.cat([predlist, predictions.view(-1).cpu()])
                labellist = torch.cat([labellist, labels.view(-1).cpu()])

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_class_map[classes[label]] += 1
                    pred_class_map[classes[label]] += 1

        print("TEST ACCURACY \n")
        class_accuracies, aggregate_accuracy = getAccuracy(correct, correct_class_map, total, pred_class_map)

        # print("\n")
        # print("CONFUSION MATRIX \n")
        # showConfusionMatrix(classes, labellist, predlist)

        print("\n")
        print("CLASSIFICATION REPORT \n")
        print(classification_report(labellist, predlist, target_names=classes))

        precisions, recalls, fscores, _ = precision_recall_fscore_support(labellist, predlist)

        for i, target_class in enumerate(classes):
            results[target_class]['accuracy'].append(class_accuracies[i])
            results[target_class]['precision'].append(precisions[i])
            results[target_class]['recall'].append(recalls[i])
            results[target_class]['fscore'].append(fscores[i])
        results['aggregate_accuracy'].append(aggregate_accuracy)

    print("<-------------------END OF TRAINING-------------------->")
    for i, target_class in enumerate(classes):
        print("\n")
        print("For class:", target_class)
        print(f'Average accuracy: {np.mean(results[target_class]["accuracy"]):.4f}'
              f'\nAverage precision: {np.mean(results[target_class]["precision"]):.4f}'
              f'\nAverage recall: {np.mean(results[target_class]["recall"]):.4f}'
              f'\nAverage F-score: {np.mean(results[target_class]["fscore"]):.4f}')

    print(f"Average accuracy: {np.mean(results['aggregate_accuracy']): .4f}")

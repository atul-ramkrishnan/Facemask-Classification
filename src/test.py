import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from pathlib import Path
import pandas as pd


def test(model, dataloader, classes, device, model_path):
    """
    Loads the saved model from the directory and outputs performance metrics on the test-set
    that include accuracy, precision, recall, F1-measure on each of the 5 classes.
    Further, it displays the confusion matrix.

    Parameters
    ----------
    model: PyTorch model
        PyTorch model used to perform the test
    dataloader: PyTorch Dataloader
        Dataloader for the test-set
    classes: list
        List of the target classes
    device: torch.device
        Device on which the Pytorch model is to be stored (CPU/GPU)

    Returns
    -------
    None
    """
    print("STARTING TEST...")
    print("==================================================")
    torch.manual_seed(0)
    model.load_state_dict(torch.load(Path(model_path), map_location='cpu'))
    model.to(device)
    correct_class_map = {label: 0 for label in classes}
    pred_class_map = {label: 0 for label in classes}
    correct = 0
    total = 0

    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    labellist = torch.zeros(0, dtype=torch.long, device='cpu')
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
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
    getAccuracy(correct, correct_class_map, total, pred_class_map)

    print("\n")
    print("CONFUSION MATRIX \n")
    showConfusionMatrix(classes, labellist, predlist)

    print("\n")
    print("CLASSIFICATION REPORT \n")
    print(classification_report(labellist, predlist, target_names=classes))


def getAccuracy(correct, correct_pred, total, total_pred):
    """
    Prints the accuracy for each target class

    Parameters
    ----------
    correct: int
        Total number of correct predictions
    correct_pred: dict
        Number of correct predictions in each target class
    total: int
        Total number of predictions
    total_pred: dict
        Number of predictions in each target class

    Returns
    -------
    class_accuracies: list
    aggregate_accuracy: float
    """
    class_accuracies = []
    print(F"Accuracy on the {total} test images: {100 * correct / total}\n")
    for label, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[label]
        class_accuracies.append(accuracy)
        print("Accuracy for class {:5s} is: {:.1f} %".format(label, accuracy))

    return class_accuracies, (100 * correct / total)


def showConfusionMatrix(classes, labels, predlist):
    """
    Prints the confusion matrix

    Parameters
    ----------
    classes: list
        List of the target classes
    labels: torch.Tensor
        A tensor of the correct labels
    predlist: torch.Tensor
        A tensor of the predicted labels

    Returns
    -------
    None
    """
    conf_matrix = confusion_matrix(labels.numpy(), predlist.numpy())
    df_cm = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(df_cm, "\n")
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap="Greens")
    plt.show()

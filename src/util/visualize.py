import numpy as np
import matplotlib.pyplot as plt


def plot_training_graphs(epochs, train_loss_list, val_loss_list, accuracy_list):
    """
    Plots the training and validation losses as well as the validation accuracies over epochs.

    Parameters
    ----------
    epochs: int
        Number of epochs the model was trained for
    train_loss_list: list
        List of training losses
    val_loss_list: list
        List of validation losses
    accuracy_list: list
        List of accuracies over epochs

    Returns
    -------
    None

    """
    epoch_list = [epoch for epoch in range(1, epochs + 1)]
    plt.plot(epoch_list, train_loss_list,
             epoch_list, val_loss_list)

    plt.title('Training and validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()

    plt.plot(epoch_list, accuracy_list)
    plt.title('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['val_accuracy'])
    plt.show()


def plot(img, target, predicted):
    """
    Display an image using the matplotlib module

    Parameters
    ----------
    img: torch.Tensor
        Tensor of the input image
    target: torch.Tensor
        Target class
    predicted: torch.Tensor
        Predicted class

    Returns
    -------
    None

    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.title("Ground Truth vs Prediction")
    plt.xlabel("Predicted: " + predicted + "\n" + "Target:" + target)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

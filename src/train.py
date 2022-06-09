import torch
from util.visualize import plot_training_graphs
from util.early_stopping import EarlyStopping


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
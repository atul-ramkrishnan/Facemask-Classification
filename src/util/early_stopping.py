import numpy as np
import torch


class EarlyStopping:
    """
    Implements early stopping mechanism.
    Saves the model as long as the validation score is decreasing.
    Stops the training if the validation score does not improve after
    a given patience.

    Parameters
    ----------
    patience: int
            Given number of attempts after validation loss
            has stopped improving before stopping training.
    save_path: str
            Path where the model should be saved.
    min_delta: float, optional
            The least amount by which the new validation loss
            should be less than the previous best validation loss
            to be counted as an improvement.

    """

    def __init__(self, patience, save_path, min_delta=0.01):
        """
        Initializes the class
        """
        self.patience = patience
        self.save_path = save_path
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.counter = 0

    def __call__(self, model, val_loss):
        """
        Compares the validation loss to the previous best validation loss
        and if we match the criteria for stopping the training.

        Parameters
        ----------
        model: torch.nn.Module
                PyTorch model that we want to save.
        val_loss: float
                Validation loss to be compared
                against previous best validation loss.

        Returns
        -------
        bool
            Whether we match the criteria for early stopping or not.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.__save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

    def __save_checkpoint(self, model):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        model: torch.nn.Module
                PyTorch model to be saved.

        Returns
        -------
        None
        """
        torch.save(model.state_dict(), self.save_path)

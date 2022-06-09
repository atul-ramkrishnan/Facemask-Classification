import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from util.get_dataloaders import DATA_TRANSFORM
from util.visualize import  plot
from pathlib import Path


def demo(model, classes, device, model_path, demo_dir):
    """
    Use the model to get the predictions on input images.

    Parameters
    ----------
    model: PyTorch model
        PyTorch model that is to be used for inference.
    classes: list
        List of the target classes
    device: torch.device
        Device on which the Pytorch model is to be stored (CPU/GPU)
    model_path: str
        Path at which the model is stored
    demo_dir: str
        Path at which the data for the demo is stored

    Returns
    -------
    None
    """
    model.load_state_dict(torch.load(Path(model_path), map_location='cpu'))
    model.to(device)

    dataset = datasets.ImageFolder(demo_dir, transform=DATA_TRANSFORM)
    batch_size = 1
    demo_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(demo_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            inputs = inputs.squeeze(0)
            plot(inputs, classes[labels], classes[predictions])

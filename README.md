# COMP 6721 Applied Artificial Intelligence -- Mask classification using Convolutional Neural Network


## Directory structure
```
├── Dataset
│  ├── Cloth-Mask
│  │  ├── 0.jpg
│  │  ├── 1.jpg
│  │  ├── ...
│  ├── FFP2-Mask
│  │  ├── 0.jpg
│  │  ├── 1.jpg
│  │  ├── ...
│  ├── Incorrectly-Worn-Mask
│  │  ├── 00000_Mask_Mouth_Chin.jpg
│  │  ├── 00001_Mask_Chin.jpg
│  │  ├── ...
│  ├── No-Mask
│  │  ├── 0688.jpg
│  │  ├── 0690.jpg
│  │  ├── ...
│  └── Surgical-Mask
│      ├── 0.jpg
│      ├── 1.jpg
│      ├── ...
├── demo
│  ├── Cloth-Mask
│  │  ├── 0.jpg
│  │  └── 1.jpg
│  ├── FFP2-Mask
│  │  ├── 397.jpg
│  │  └── 399.jpg
│  ├── Incorrectly-Worn-Mask
│  │  ├── 00000_Mask_Mouth_Chin.jpg
│  │  └── 00001_Mask_Chin.jpg
│  ├── No-Mask
│  │  ├── 3.jpg
│  │  └── 9.jpg
│  └── Surgical-Mask
│      ├── 0.jpg
│      └── 1.jpg
├── saved_models
│  └── SimpleCNN.pth
└── src
    ├── config.py
    ├── demo.py
    ├── main.py
    ├── models
    │  ├── models.py
    │
    ├── test.py
    ├── train.py
    └── util
        ├── early_stopping.py
        ├── get_dataloaders.py
        └── visualize.py
```

Dataset -- Contains the dataset used for training/testing/validation

demo -- Contains the image samples used for demonstration

saved_models -- Contains the models that are saved during training

src/config.py -- Contains configuration items such as learning_rate, batch_size, etc

src/demo.py -- Contains code for displaying ground truth and predicted label

src/main.py -- Main function

src/models/models.py -- Contains three models that were compared

src/test.py -- Contains code for obtaining performance metrics and for displaying the confusion matrix

src/util/early_stopping.py -- Contains code for early stopping mechanism

src/util_get_dataloaders.py -- Contains code for creating the train/val/test dataloaders

src/util/visualize.py -- Contains code for plotting various graphs and for displaying images


## Training

To train the model with the default options, you can simply run the following command from within src folder --
```bash
python main.py train BaselineCNN
```
This will train the first model 'BaselineCNN'. The model with the lowest validation loss will be saved in the saved_models folder which can be then used for testing and inference.

Other models can be trained by replacing 'BaselineCNN' with any other model in models.py.

Configuration options such as batch_size, learning_rate, and num_epochs can be changed by changing the values in config.py.

## Testing
To test the 'BaselineCNN' model, you can run the following command from within the src folder --
```bash
python main.py test BaselineCNN
```
This will fetch performance metrics and generate a confusion matrix for all the target classes.

Other models can be tested by replacing 'BaselineCNN' with any other model in models.py

## Inference
You can compare the ground truths and predicted values for any image by running the following command from within the src folder --
```bash
python main.py demo BaselineCNN
```
Other models can be used by replacing 'BaselineCNN' with any other model in models.py

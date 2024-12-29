# EOLVision
EOLVision is an End Of Line Testing CNN Model
## Project Overview

This project involves the development and implementation of a computer vision model for classifying images into various categories. The model is trained using Convolutional Neural Networks (CNNs) and various data preprocessing and augmentation techniques to improve accuracy and robustness.

## Installation and Execution

1. **Install dependencies**:
   Make sure you have the necessary Python libraries installed. You can install the dependencies with pip:
   ```bash
   pip install -r requirements.txt
   ```
   or alternatively with conda:

    ```bash
   conda env create -f environment.yml
   conda activate end-of-line-testing
   ```

3. **Train and or the model**:
    Run the main script project.py to train a model with the configuration set in `build_model.py` and analyze the results or alternatively load a model. If a model with the name [model_name] already exists under model/ it will be loaded, otherwise it will be trained.
   
    ```bash
   python eolvision.py [model_name]
   ```

    The model provided in this repo is called Model so simply load it by running
    ```bash
   python eolvision.py Model
   ```

## Files and Functions

### `get_data.py`

This file contains functions for loading and preprocessing image data.

- `retrieve_data(data_dir, labels)`: Loads and processes image data from a specified directory.
- `get_data(labels)`: Loads and processes the train, validation and test data.

### `build_model.py`

This file contains functions for creating, training, and saving the model.

- `set_seed(seed)`: Sets the random seed for reproducibility.
- `add_layers(model, input_shape, num_classes)`: Builds a Sequential model with specified layers.
- `build_model(x_train, y_train, x_val, y_val, labels, name)`: Creates, trains and saves a model.
- `save_model(model, history, name)`: Saves the model and its training history.
- `load_model(name)`: Loads the model and its training history.

### `analyse_data.py`

This file contains functions for analyzing and visualizing the model's performance.

- `plot_loss(history, ax)`: Plots training and validation loss.
- `plot_accuracy(history, ax)`: Plots training and validation accuracy.
- `plot_model(history)`: Plots training and validation loss and accuracy.
- `prepare_evaluation_data(x_true, y_true, labels, model)`: Formats the validation data and model predictions for evaluation.
- `print_performance(model, x_train, y_train, x_val, y_val, labels, x_test = None, y_test = None)`: Prints the performance of the model on training, validation and optionally test data.
- `visualize_predictions(model, x_true, y_true, labels)`: Visualizes the model's predictions on validation data.
- `print_confusion_matrix(model, x_true, y_true, labels)`: Plots the confusion matrix.

### `project.py`

This file is the main script that loads the data, trains the model, and analyzes the results.

- Loads and preprocesses the training and validation data.
- Checks if a model already exists and loads it or trains a new model.
- Prints the model summary.
- Analyzes the model's performance and visualizes the predictions.


## Results
The results of the model will be output to the console and the visualizations will be displayed. The trained model and its training history will be saved in the models and history directories, respectively.

## Contact
For questions or suggestions, you can reach me at inf22131@lehre.dhbw-stuttgart.de.


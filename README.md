# End Of Line Testing CNN Model

## Project Overview

This project involves the development and implementation of a computer vision model for classifying images into various categories. The model is trained using Convolutional Neural Networks (CNNs) and various data preprocessing and augmentation techniques to improve accuracy and robustness.

## Files and Functions

### `get_data.py`

This file contains functions for loading and preprocessing image data.

- `retrieve_data(data_dir, labels)`: Loads and processes image data from a specified directory.
- `train_val_split(x, y)`: Splits the data into training and validation sets.
- `get_train_val(labels)`: Loads and processes the training and validation data.
- `get_test(labels)`: Loads and processes the test data.

### `build_model.py`

This file contains functions for creating, training, and saving the model.

- `set_seed(seed)`: Sets the random seed for reproducibility.
- `add_layers(model, input_shape, num_classes)`: Builds a Sequential model with specified layers.
- `build_model(x_train, y_train, x_val, y_val, labels, name)`: Creates, compiles, and trains a model.
- `save_model(model, history, name)`: Saves the model and its training history.
- `load_model(name)`: Loads the model and its training history.
- `create_data(x_train, y_train, labels)`: Generates more training data for each class through data augmentation.

### `analyse_data.py`

This file contains functions for analyzing and visualizing the model's performance.

- `plot_model(history)`: Plots training and validation loss and accuracy.
- `format_data(x_val, y_val, labels, model)`: Formats the validation data and model predictions for evaluation.
- `print_performance(model, y_val, y_train, x_val, x_train, labels)`: Prints the performance of the model on training and validation data.
- `visualize_predictions(model, x_val, y_val, labels)`: Visualizes the model's predictions on validation data.

### `project.py`

This file is the main script that loads the data, trains the model, and analyzes the results.

- Loads and preprocesses the training and validation data.
- Checks if a model already exists and loads it or trains a new model.
- Prints the model summary.
- Analyzes the model's performance and visualizes the predictions.

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


2. **Prepare the data**:
    Place the training and test data in the appropriate directories under data.

3. **Train and run the model**:
    Run the main script project.py to train the model and analyze the results
    ```bash
   python project.py
   ```

## Results
The results of the model will be output to the console and the visualizations will be displayed. The trained model and its training history will be saved in the models and history directories, respectively.

## Contact
For questions or suggestions, you can reach me at inf22131@lehre.dhbw-stuttgart.de.


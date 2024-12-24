# Import necessary functions and modules
from get_data import get_train_val 
from build_model import load_model, build_model
from analyse_data import plot_model
from analyse_data import print_performance
from analyse_data import visualize_predictions
import os  

# Load and preprocess the training and validation data
labels = ['blue', 'fail', 'red', 'white']
x_train, y_train, x_val, y_val = get_train_val(labels=labels)

# Define the model name
name = 'Model18'

# Define the base directory and file path for saving/loading the model
base_dir = os.getcwd()
file_path = os.path.join(base_dir, "models", f"{name}.keras")

# Check if the model already exists
if os.path.isfile(file_path):
    # Load the existing model and its training history
    model, history = load_model(name)
else:
    # Build, compile, and train a new model
    model, history = build_model(x_train, y_train, x_val, y_val, labels, name)
    history = history.history

# Print the model summary
model.summary()

# Print the performance of the model on training and validation data
print_performance(model, y_val, y_train, x_val, x_train, labels)

# Visualize the model's predictions on validation data
visualize_predictions(model, x_val, y_val, labels)

# Plot the training and validation loss and accuracy
plot_model(history)
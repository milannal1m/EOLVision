# Import necessary functions and modules
from get_data import get_train_val 
from build_model import load_model, build_model
from analyse_data import plot_model
from analyse_data import print_performance
from analyse_data import visualize_predictions
import os 
import sys

labels = ['blue', 'fail', 'red', 'white']
x_train, y_train, x_val, y_val = get_train_val(labels=labels)

name = sys.argv[1]


base_dir = os.getcwd()
file_path = os.path.join(base_dir, "models", f"{name}.keras")

if os.path.isfile(file_path):
    model, history = load_model(name)
else:
    model, history = build_model(x_train, y_train, x_val, y_val, labels, name)
    history = history.history

model.summary()

print_performance(model, y_val, y_train, x_val, x_train, labels)

visualize_predictions(model, x_val, y_val, labels)

plot_model(history)
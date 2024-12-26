from get_data import get_data
from build_model import load_model, build_model
from analyse_data import plot_model
from analyse_data import print_performance
from analyse_data import visualize_predictions
from analyse_data import print_confusion_matrix
import os 
import sys

labels = ['blue', 'fail', 'red', 'white']
x_train, y_train, x_val, y_val, x_test, y_test = get_data(labels)

name = sys.argv[1]


base_dir = os.getcwd()
file_path = os.path.join(base_dir, "models", f"{name}.keras")

if os.path.isfile(file_path):
    model, history = load_model(name)
else:
    model, history = build_model(x_train, y_train, x_val, y_val, labels, name)
    history = history.history

model.summary()

print_performance(model, x_train, y_train, x_val, y_val, labels)

print_confusion_matrix(model, x_val, y_val, labels)

visualize_predictions(model, x_val, y_val, labels)

plot_model(history)
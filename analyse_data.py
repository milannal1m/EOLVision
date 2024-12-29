import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import random

def plot_model(history):
    """
    Plots training and validation loss and accuracy from a Keras History object.

    Args:
        history: Keras History object containing loss, val_loss, accuracy, and val_accuracy.
    """
    
    def plot_loss(history, ax):
        """
        Plots training and validation loss from a Keras History object.

        Args:
            history: Keras History object containing loss and val_loss.
            ax: The subplot axis to plot on.
        """
        ax.plot(history['loss'], 'r-', label='Training Loss')
        ax.plot(history['val_loss'], 'g-', label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Training Loss', 'Validation Loss'], loc='upper right')

    def plot_accuracy(history, ax):
        """
        Plots training and validation accuracy from a Keras History object.

        Args:
            history: Keras History object containing accuracy and val_accuracy.
            ax: The subplot axis to plot on.
        """
        ax.plot(history['accuracy'], 'r-', label='Training Accuracy')
        ax.plot(history['val_accuracy'], 'g-', label='Validation Accuracy')
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_loss(history, axes[0])
    plot_accuracy(history, axes[1])

    plt.tight_layout()
    plt.show()

def format_data(x_val, y_val, labels, model):
    """
    Formats the validation data and model predictions for evaluation.

    Args:
        x_val: Validation data features.
        y_val: Validation data labels.
        labels: List of label names.
        model: Trained Keras model.

    Returns:
        y_pred_string: Predicted labels as strings.
        y_val_string: True labels as strings.
    """
    label_binarizer = sklearn.preprocessing.LabelBinarizer()

    y_pred_prob = model.predict(x_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    label_binarizer.fit([0, 1, 2, 3])
    y_pred = label_binarizer.transform(y_pred)

    label_binarizer.fit(labels)
    y_val_string = label_binarizer.inverse_transform(y_val)
    y_pred_string = label_binarizer.inverse_transform(y_pred)

    return y_pred_string, y_val_string

def print_performance(model, x_train, y_train, x_val, y_val, labels, x_test = None, y_test = None):
    """
    Prints the performance of the model on training, validation and optionally test data.

    Args:
        model: Trained Keras model.
        x_train: Training data features.
        y_train: Training data labels.
        x_val: Validation data features.
        y_val: Validation data labels.
        labels: List of label names.
        x_test: Test data features.
        y_test: Test data labels.

    Optional Args:
        x_test: Test data features.
        y_test: Test data labels
    """
    train_score = model.evaluate(x_train, y_train, verbose=0)
    val_score = model.evaluate(x_val, y_val, verbose=0)

    print(f"\nTraining accuracy: {train_score[1]:.3f}")
    print(f"Validation accuracy: {val_score[1]:.3f}\n")

    x_metric = x_val
    y_metric = y_val

    if x_test is not None and y_test is not None:
        test_score = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_score[1]:.3f}\n")
        x_metric = x_test
        y_metric = y_test

    y_pred_string, y_metric_string = format_data(x_metric, y_metric, labels, model)

    print(sklearn.metrics.classification_report(y_metric_string, y_pred_string))

def visualize_predictions(model, x_true, y_true, labels):
    """
    Visualizes the model's predictions on validation data.

    Args:
        model: Trained Keras model.
        x_true: True data features.
        y_true: True data labels.
        labels: List of label names.
    """
    i = 0
    num_examples = 4
    prop_class = []
    mis_class = []

    y_pred_prob = model.predict(x_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = [np.argmax(row) for row in y_true]

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            prop_class.append(i)
        if y_true[i] != y_pred[i]:
            mis_class.append(i)

    random.shuffle(prop_class)
    random.shuffle(mis_class)

    prop_class = prop_class[:num_examples]
    mis_class = mis_class[:num_examples]
    
    count = 0
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(12, 6)

    for i in range(4):
        ax[0, i].imshow(x_true[prop_class[count]])
        ax[0, i].set_title(f"Predicted: {labels[y_pred[prop_class[count]]]}\nActual: {labels[y_true[prop_class[count]]]}")
        ax[0, i].axis('off')
        count += 1

    count = 0

    for i in range(4):
        ax[1, i].imshow(x_true[mis_class[count]])
        ax[1, i].set_title(f"Predicted: {labels[y_pred[mis_class[count]]]}\nActual: {labels[y_true[mis_class[count]]]}")
        ax[1, i].axis('off')
        count += 1

def print_confusion_matrix(model, x_true, y_true, labels):
    """
    Plots the confusion matrix for the given model predictions and true labels.

    Arguments:
    model -- the trained Keras model used for making predictions.
    x_true -- numpy array of true data features.
    y_true -- numpy array of true data labels.
    labels -- list of label names for classification.

    This function predicts the labels for the given true data features using the provided model,
    computes the confusion matrix, and plots it as a heatmap.
    """
    y_pred_prob = model.predict(x_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = [np.argmax(row) for row in y_true]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
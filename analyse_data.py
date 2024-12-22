import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import random

def plot_model(history):
    
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
        ax.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')


    # Erstelle Subplots nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 1 Reihe, 2 Spalten

    # Plots zeichnen
    plot_loss(history, axes[0])
    plot_accuracy(history, axes[1])

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

def format_data(x_val, y_val, labels, model):
    label_binarizer = sklearn.preprocessing.LabelBinarizer()

    y_pred_prob = model.predict(x_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    label_binarizer.fit([0, 1, 2, 3])
    y_pred = label_binarizer.transform(y_pred)

    label_binarizer.fit(labels)
    y_val_string = label_binarizer.inverse_transform(y_val)
    y_pred_string = label_binarizer.inverse_transform(y_pred)

    return y_pred_string, y_val_string

def print_performance(model, y_val, y_train, x_val, x_train, labels):

    train_score = model.evaluate(x_train, y_train,verbose = 0)
    val_score = model.evaluate(x_val, y_val,verbose = 0)

    print(f"Training accuracy: {train_score[1]:.2f}")
    print(f"Validation accuracy: {val_score[1]:.2f}")

    y_pred_string , y_val_string = format_data(x_val, y_val, labels, model)

    print(sklearn.metrics.classification_report(y_val_string, y_pred_string))


def visualize_predictions(model, x_val, y_val, labels):
    i = 0
    num_examples = 4
    prop_class = []
    mis_class = []

    y_pred_prob = model.predict(x_val)
    y_pred = np.argmax(y_pred_prob, axis=1)

    y_val = [np.argmax(row) for row in y_val]

    for i in range(len(y_val)):
        if y_val[i] == y_pred[i]:
            prop_class.append(i)
        if y_val[i] != y_pred[i]:
            mis_class.append(i)

    random.shuffle(prop_class)
    random.shuffle(mis_class)

    prop_class = prop_class[:num_examples]
    mis_class = mis_class[:num_examples]
    
    count = 0
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(12, 6)

    for i in range(4):
        ax[0, i].imshow(x_val[prop_class[count]])
        ax[0, i].set_title(f"Predicted: {labels[y_pred[prop_class[count]]]}\nActual: {labels[y_val[prop_class[count]]]}")
        ax[0, i].axis('off')
        count += 1

    count = 0

    for i in range(4):
        ax[1, i].imshow(x_val[mis_class[count]])
        ax[1, i].set_title(f"Predicted: {labels[y_pred[mis_class[count]]]}\nActual: {labels[y_val[mis_class[count]]]}")
        ax[1, i].axis('off')
        count += 1



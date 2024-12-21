import matplotlib.pyplot as plt

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
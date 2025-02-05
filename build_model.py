import os 
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import random
import pickle

def set_seed(seed=0):
    """
    Sets the random seed for reproducibility across various libraries and environments.

    Arguments:
    seed -- integer, the seed value to be set for reproducibility (default is 0).
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.utils.set_random_seed(seed)

def add_layers(model, input_shape, num_classes):
    """
    Builds a Sequential model with specified layers for image classification.

    Arguments:
    model -- Base model to be extended
    input_shape -- The shape of the images in the dataset.
    num_classes -- The number of classes for classification.

    Returns:
    model -- a Sequential model built as per the specified architecture.
    """
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.40))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model

def save_model(model,history,name):
    """
    Saves the model and its training history to the specified directory.

    Arguments:
    model -- the trained Keras model to be saved.
    history -- training history of the model.
    name -- string, name of the model for saving purposes.
    """
    model.save('model/'+name+'.keras')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with tf.io.gfile.GFile('model/model.tflite', 'wb') as f:
        f.write(tflite_model)

    with open('history/'+name+'_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

def load_model(name):
    """
    Loads the model and its training history from the specified directory
    
    Arguments:
    name -- string, name of the model to be loaded.

    Returns:
    model -- the loaded Keras model.
    history -- training history of the model.
    """
    model = tf.keras.models.load_model('model/'+name+'.keras')

    with open('history/'+name+'_history.pkl', 'rb') as f:
        history= pickle.load(f)
        
    return model,history


def build_model(x_train,y_train,x_val,y_val,labels,name):
    """
    Builds, compiles, and trains a Keras Sequential model with specified layers for image classification.

    Arguments:
    x_train -- numpy array of training data features.
    y_train -- numpy array of training data labels.
    x_val -- numpy array of validation data features.
    y_val -- numpy array of validation data labels.
    labels -- list of label names for classification.
    name -- string, name of the model for saving purposes.

    Returns:
    model -- the trained Keras model.
    history -- training history of the model.
    """
    model = tf.keras.models.Sequential()
    model = add_layers(model, x_train.shape[1:], len(labels))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0004), loss=loss_fn, metrics=['accuracy'])
    set_seed()

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,restore_best_weights=True)
    lr_reduction_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience=3, verbose=1, factor=0.7, min_lr=0.000001)

    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[lr_reduction_on_plateau, early_stopper], batch_size=32)
    save_model(model, history, name)

    return model, history
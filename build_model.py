import os 
import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
import random
import pickle

def set_seed(seed=0):
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
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.20))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(0.025)))  
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.30))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    

    return model
    
def save_model(model,history,name):
    """
    Saves the model to the specified directory.
    """
    model.save('models/'+name+'.keras')

    with open('history/'+name+'_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)



def load_model(name):
    """
    Loads the model from the specified directory.
    """

    model = tf.keras.models.load_model('models/'+name+'.keras')
    with open('history/'+name+'_history.pkl', 'rb') as f:
        history= pickle.load(f)
    return model,history

def build_model(x_train,y_train,x_val,y_val,labels,name):
    """
    Builds a Sequential model with specified layers for image classification.
    Returns:
    model -- a Sequential model built as per the specified architecture.
    """

    model = tf.keras.models.Sequential()
    model = add_layers(model, x_train.shape[1:], len(labels))

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006), loss=loss_fn, metrics=['accuracy'])
    set_seed()

    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # don't randomly flip images vertically

    # Fit the augmentation model to the data
    datagen.fit(x_train)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15,restore_best_weights=True)
    lr_reduction_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience=3, verbose=1, factor=0.7, min_lr=0.000001)

    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[lr_reduction_on_plateau, early_stopper], batch_size=64)
    save_model(model, history, name)

    return model, history
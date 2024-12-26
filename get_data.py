import os
import numpy as np
from PIL import Image
import sklearn.model_selection
import sklearn.preprocessing

def retrieve_data(data_dir, labels):
    """
    This function loads and preprocesses image data from a specified directory.

    Arguments:
    data_dir -- the directory path containing image data. The directory is expected to have subdirectories
                named after the labels/classes, with each subdirectory containing respective images.
    labels -- a list of labels/classes corresponding to the subdirectories in data_dir.

    Returns:
    A tuple of two numpy arrays: the preprocessed images and the corresponding labels.
    """
    x = []
    y = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            if img.lower().endswith(valid_extensions):
                image = Image.open(os.path.join(path, img))
                resized_img = image.resize((320, 240))
                resized_arr = np.array(resized_img)

            x.append(resized_arr)
            y.append(class_num)
    return (np.array(x, dtype=int), np.array(y, dtype=int))

def get_data(labels):
    """
    Loads and preprocesses the training, validation, and test data.

    Arguments:
    labels -- a list of labels/classes

    Returns:
    A tuple containing the training, validation, and test sets: (x_train, y_train, x_val, y_val, x_test, y_test).
    """

    x_train, y_train = retrieve_data("data/data/train/train_four_label/", labels)
    x_test_val, y_test_val = retrieve_data("data/data/test/test_four_label/", labels)

    x_test, x_val, y_test, y_val = sklearn.model_selection.train_test_split(
        x_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=0)

    x_train = x_train / 255
    x_val = x_val / 255
    x_test = x_test / 255

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_val = label_binarizer.transform(y_val)
    y_test = label_binarizer.transform(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    """
    Main function to load and preprocess the data, and print the shapes of the datasets.
    """

    labels = ['blue', 'fail', 'red', 'white']

    x_train, y_train, x_val, y_val, x_test, y_test = get_data(labels)

    print("\nTraining: " + str(sum([len(y_train)])) + " Shape X: " + str(x_train.shape) + " Shape Y: " + str(y_train.shape) + "\n")
    print("Validation: " + str(sum([len(y_val)]))+ " Shape X: " + str(x_val.shape) + " Shape Y: " + str(y_val.shape) + "\n")
    print("Test: " + str(sum([len(y_test)])) + " Shape X: " + str(x_test.shape) + " Shape Y: " + str(y_test.shape) + "\n")
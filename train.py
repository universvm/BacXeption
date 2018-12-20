import os
import numpy as np

from keras.callbacks import CSVLogger
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from skimage import transform
from skimage.io import imread

from bac_detection.image_processing import thresholding_img
from bac_detection.image_processing import preprocess_cutout
from neural_network.network_structure import build_network


def load_data(data_dir):
    # TODO: variable in config
    class_list = ['acceptable', 'discarded']

    # Main Input Lists:
    images = []
    labels = []

    # looping twice (flat and not_flat):
    for c in range(len(class_list)):

        # Go through directory (either flat or not_flat:
        for (dirpath, dirnames, filenames) in os.walk \
                (data_dir + class_list[c] + '/'):

            print(dirpath, dirnames, filenames)
            # Loop through images:
            for i in range(len(filenames)):
                # TODO: variable in config
                if filenames[i].endswith(".tiff"):

                    # Reading image:
                    current_bac = imread(dirpath + filenames[i], as_gray=True)
                    thresh_image = thresholding_img(current_bac)

                    # Resize multiplier:
                    thresh_image = transform.resize(thresh_image, (180, 180, 1))

                    images.append(thresh_image)
                    labels.append(c)

    return images, labels


def split_data(images, labels):
    """ Divide the data into train and test set """
    # TODO: variable in config
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test


def augment_data(X_train, y_train):
    """

    :param X_train: 4d np array with normalized cutouts
    :param y_train: python array with int 0 or 1
    :return X_train: augmented X_train 8x length
    :return y_train: augmented y_train 8x length
    """

    print("Current number of training images is " + str(len(X_train)))
    print("Length of y_train " + str(len(y_train)))

    # Add rotations:
    for i in range(len(X_train)):
        for rot in range(1, 4):
            rotated = np.rot90(X_train[i, :, :, :], rot)

            X_train = np.append(X_train, np.asarray([rotated]), axis=0)
            y_train.append(y_train[i])

    # Add flip:
    for i in range(len(X_train)):
        fliplr_bac = np.fliplr(X_train[i, :, :, :])
        flipud_bac = np.flipud(X_train[i, :, :, :])

        X_train = np.append(X_train, np.asarray([fliplr_bac]), axis=0)
        y_train.append(y_train[i])
        X_train = np.append(X_train, np.asarray([flipud_bac]), axis=0)
        y_train.append(y_train[i])

    print("The current amount of training images is " + str(len(X_train)))
    print("Length of y_train " + str(len(y_train)))

    return X_train, y_train


if __name__ == "__main__":
    # Define data directory:
    data_dir = "data/"
    epochs = 300

    print("Data directory is " + data_dir)

    # Load data:
    print("Loading data...")
    images, labels = load_data(data_dir)
    print("Data loaded successfully.")

    # Reshape:
    images = preprocess_cutout(images)

    # Split data: (before augmentation to avoid bias)
    print('Splitting in train and test data...')
    X_train, X_test, y_train, y_test = split_data(images, labels)

    # Normalize data:
    X_train, y_train = augment_data(X_train, y_train)
    X_test, y_test = augment_data(X_test, y_test)
    # TODO: refactor:
    # 1-hot encoding:
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Create CNN model
    print("Building CNN...")
    model = build_network()

    # Test model:
    keras_logger = CSVLogger('log.csv', append=True, separator=',')
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=100, callbacks=[keras_logger])

    # Final evaluation of the model:
    scores = model.evaluate(X_test, y_test, verbose=1)

    print("Accuracy: {0}".format(scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("neural_network/models/model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("neural_network/models/model.h5")

    print("Exit.")


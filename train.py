
# from skimage.viewer import ImageViewer
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values

#   Sci-Kit Learn: (for splitting train data)
from sklearn.model_selection import train_test_split
#   Processing:
import os
import numpy as np
from scipy.ndimage import imread
from skimage import io
from skimage import exposure
# Test:
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage import transform
from google.colab import files


def load_data(data_dir):

    status_list = ['acceptable', 'discarded']

    # Main Input Lists:
    images = []
    labels = []

    # looping twice (flat and not_flat):
    for f in range(len(status_list)):

        # Go through directory (either flat or not_flat:
        for (dirpath, dirnames, filenames) in os.walk \
                (data_dir + status_list[f] + '/'):

            print(dirpath, dirnames, filenames)
            # Loop through images:
            for i in range(len(filenames)):
                if filenames[i].startswith("."):
                    pass

                # If file is an image:
                elif filenames[i].endswith(".tiff"):

                    # Reading image:
                    current_bac = imread(dirpath + filenames[i], flatten=False)

                    # Normalize:
                    current_bac = exposure.rescale_intensity(current_bac, in_range=(
                        current_bac.min(), current_bac.max()))

                    # current_bac = current_bac / np.max(current_bac)

                    # Resize multiplier:
                    current_bac = transform.resize(current_bac, (180, 180, 3))

                    # mltpl = 180 // max(current_bac.shape)

                    # current_bac = transform.resize(current_bac,
                    #                                (current_bac.shape[0]*mltpl, current_bac.shape[1]*mltpl))
                    # Anti-Aliasing only available in 0.14 scikit image

                    # current_bac = np.pad(current_bac, ((0, (180-current_bac.shape[0])),
                    #                                    (0, (180-current_bac.shape[1]))), "constant")

                    # Show bacteria
                    # plt.imshow(current_bac)
                    # plt.show()
                    # plt.pause(0.01)
                    # plt.clf()

                    # Appending lists:
                    images.append(current_bac)
                    labels.append(f)

                else:
                    continue

    return images, labels

def split_data(images, labels):
    """ Divide the data into train and test set """

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test


def augmentor_data(X_train, y_train):
    """ Increase amount of data by varying the orientation of bacteria """

    print("The current number of training images is " + str(len(X_train)))
    print("Length of y_train " + str(len(y_train)))

    # Add rotations:
    for i in range(len(X_train)):
        for rot in range(1, 4):
            # plt.ion()

            rotated = np.rot90(X_train[i, :, :, :], rot)
            # plt.imshow(rotated.reshape(180, 180))
            # plt.show()

            # Append:
            X_train = np.append(X_train, np.asarray([rotated]), axis=0)
            y_train.append(y_train[i])
            print(len(X_train))
            print(len(y_train))

    # Add flip:
    for i in range(len(X_train)):

        # Flip:
        fliplr_bac = np.fliplr(X_train[i, :, :, :])
        flipud_bac = np.flipud(X_train[i, :, :, :])

        # Append:
        X_train = np.append(X_train, np.asarray([fliplr_bac]), axis=0)
        y_train.append(y_train[i])
        X_train = np.append(X_train, np.asarray([flipud_bac]), axis=0)
        y_train.append(y_train[i])

    print("The current amount of training images is " + str(len(X_train)))
    print("Length of y_train " + str(len(y_train)))

    return X_train, y_train


if __name__ == "__main__":
    # Define data directory:
    data_dir = "drive/Colab Notebooks/Pilizota/data/expanded_3/"
    print('Data directory is ' + data_dir)

    # Load data:
    print('Loading data...')
    images, labels = load_data(data_dir)
    print('Data loaded successfully.')

    # Reshape:
    images = np.array(images)
    print(images.shape)
    images = images.reshape(len(images), 180, 180, 3)

    # Split data: (before augmentation to avoid bias)
    print('Splitting the data in train and test data...')
    X_train, X_test, y_train, y_test = split_data(images, labels)
    print('Splitting successful.')

    # Normalize data:
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')

    X_train, y_train = augmentor_data(X_train, y_train)
    X_test, y_test = augmentor_data(X_test, y_test)


    # 1-hot encoding:
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print("Data normalized and 1-hot encoded successfully.")

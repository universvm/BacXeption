import keras
import numpy as np
import os

from skimage import transform
from skimage.io import imread

from bac_detection.image_processing import thresholding_img


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, id_list, id_dict, dim, batch_size=32, n_classes=2,
                 train=True, shuffle=True):

        self.dim = dim
        self.batch_size = batch_size
        self.id_dict = id_dict
        self.id_list = id_list
        self.n_channels = 1
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.id_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.id_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            y[i] = self.id_dict[ID]

            current_img = imread(os.path.join('data', str(self.id_dict[ID]),
                                              list_IDs_temp[i]),
                                 as_gray=True)
            current_img = thresholding_img(current_img)
            current_img = transform.resize(current_img, self.dim)

            X[i] = current_img

        if self.train:
            X_augmented, y_augmented = self.augment_data(X, y)

        return X_augmented, keras.utils.to_categorical(y_augmented,
                                                    num_classes=self.n_classes)

    def augment_data(self, X_train, y_train):
        """

        :param X_train: 4d np array with normalized cutouts
        :param y_train: python array with int 0 or 1
        :return X_train: augmented X_train 8x length
        :return y_train: augmented y_train 8x length
        """

        print("Current number of training images is " + str(len(X_train)))
        print("Length of y_train " + str(len(y_train)))

        # TODO: refactor for loops
        # Add rotations:
        for i in range(len(X_train)):
            for rot in range(1, 4):
                rotated = np.rot90(X_train[i, :, :, :], rot)

                X_train = np.append(X_train, np.asarray([rotated]), axis=0)
                y_train = np.append(y_train, np.asarray([y_train[i]]), axis=0)

        # Add flip:
        for i in range(len(X_train)):
            fliplr_bac = np.fliplr(X_train[i, :, :, :])
            flipud_bac = np.flipud(X_train[i, :, :, :])
            X_train = np.append(X_train, np.asarray([fliplr_bac]), axis=0)
            y_train = np.append(y_train, np.asarray([y_train[i]]), axis=0)
            X_train = np.append(X_train, np.asarray([flipud_bac]), axis=0)
            y_train = np.append(y_train, np.asarray([y_train[i]]), axis=0)

        return X_train, y_train

import os

from keras.callbacks import CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split

from neural_network.network_structure import build_network
from neural_network.image_generator import DataGenerator


def load_ids(data_dir, n_classes):

    # Main Input Lists:
    id_dict = {}  # {ID: label}

    # Main Loop:
    for c in range(n_classes):

        # Go through directory (either flat or not_flat:
        for (dirpath, dirnames, filenames) in os.walk \
                (data_dir + str(c) + '/'):

            # Loop through images:
            for i in range(len(filenames)):
                # TODO: variable in config
                if filenames[i].endswith(".tiff"):
                    id_dict[filenames[i]] = c

    return id_dict


def split_data(id_dict):
    """ Divide the data into train and test set """
    # TODO: variable in config
    X_train, X_test = train_test_split(list(id_dict.keys()), test_size=0.2)

    return X_train, X_test


if __name__ == "__main__":
    # Define data directory:
    data_dir = "data/"
    EPOCHS = 300
    INPUT_DIM = (180, 180, 1)
    BATCH_SIZE = 32
    N_CLASSES = 2

    print("Data directory is " + data_dir)

    # Load data:
    print("Loading data...")
    id_dict = load_ids(data_dir, N_CLASSES)

    # Split data: (before augmentation to avoid bias)
    print('Splitting in train and test data...')
    X_train, X_test = split_data(id_dict)

    # Reshape:
    # images = preprocess_cutout(images)
    # TODO: Call generator
    train_generator = DataGenerator(X_train, id_dict, dim=INPUT_DIM,
                                    batch_size=BATCH_SIZE,
                                    n_classes=N_CLASSES,
                                    train=True,
                                    shuffle=True)
    validation_generator = DataGenerator(X_test, id_dict, dim=INPUT_DIM,
                                         batch_size=BATCH_SIZE,
                                         n_classes=N_CLASSES,
                                         train=True,
                                         shuffle=True)

    # Create CNN model
    print("Building CNN...")
    model = build_network()

    # Loggers:
    # TODO Refactor in config
    csv_logger = CSVLogger('neural_network/models/log.csv', append=True,
                           separator=',')
    tensor_logger = TensorBoard(log_dir='neural_network/models/',
                                histogram_freq=0, write_graph=True,
                                write_images=True)

    # Test model:
    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=[csv_logger, tensor_logger])

    # serialize model to JSON
    model_json = model.to_json()
    # TODO Refactor in config
    with open("neural_network/models/model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("neural_network/models/model.h5")

    print("Exit.")


from config import *

from keras.callbacks import CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split

from neural_network.network_structure import build_network
from neural_network.image_generator import DataGenerator


def _load_ids(TRAIN_IMG_PATH, N_CLASSES):

    # Main Input Lists:
    id_dict = {}  # {ID: label}

    # Main Loop:
    for c in range(N_CLASSES):

        # Go through directory (either flat or not_flat:
        for (dirpath, dirnames, filenames) in os.walk(
                os.path.join(TRAIN_IMG_PATH, str(c))):

            # Loop through images:
            for i in range(len(filenames)):
                if filenames[i].endswith(FORMAT_IMG):
                    id_dict[filenames[i]] = c

    return id_dict


def _split_data(id_dict):
    """ Divide the data into train and test set """
    X_train, X_test = train_test_split(list(id_dict.keys()), test_size=0.2)

    return X_train, X_test


if __name__ == "__main__":
    print("Data directory is " + TRAIN_IMG_PATH)

    # Load data:
    print("Loading data...")
    id_dict = _load_ids(TRAIN_IMG_PATH, N_CLASSES)

    # Split data: (before augmentation to avoid bias)
    print("Splitting in train and test data...")
    X_train, X_test = _split_data(id_dict)

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
    csv_logger = CSVLogger(os.path.join(LOG_DIR, 'log.csv'),
                           append=True,
                           separator=',')
    tensor_logger = TensorBoard(log_dir=LOG_DIR,
                                histogram_freq=0,
                                write_graph=True,
                                write_images=True)

    # Test model:
    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=[csv_logger, tensor_logger])

    model_json = model.to_json()
    with open(os.path.join(LOG_DIR, 'model.json'), 'w') as json_file:
        json_file.write(model_json)
        model.save_weights(os.path.join(LOG_DIR, 'model.h5'))

    print("Exit.")


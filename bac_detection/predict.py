from keras.models import model_from_json


def _load_model(model_path, loss_func, optimizer, metrics):
    """
    :param model_path: string
    :return: model: keras model object
    """

    print("Loading model...")
    with open(model_path + 'model.json', 'r') as json_file:
        model_json = json_file.read()

    # Load model in Keras:
    model = model_from_json(model_json)

    # Load weights:
    model.load_weights(model_path + 'model.h5')
    model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

    return model


def predict_flatness(X_test, model_path, loss_func, optimizer, metrics):
    """ Uses the model to predict flatness. Must be given path argument.
    :param X_test: 4d numpy array with bacterial images
    :param model_path: str path to models
    :return prediction: float probability of flatness
    """

    model = _load_model(model_path, loss_func, optimizer, metrics)
    prediction = model.predict(X_test)

    return prediction

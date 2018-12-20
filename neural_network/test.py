from keras.models import model_from_json


def load_model(path):
    """
    :param path: string
    :return: model: keras model object
    """

    print("Loading model")
    # TODO: Refactor magic variable
    with open(path+'model.json', 'r') as json_file:
        model_json = json_file.read()

    # Load model in Keras:
    model = model_from_json(model_json)

    # Load weights:
    # TODO: Refactor magic variable
    model.load_weights(path+'model.h5')
    # TODO: Refactor magic variable
    model.compile(loss='mean_squared_error', optimizer='adamax', metrics=[
        'accuracy'])

    return model


def predict_flatness(X_test):
    """ Uses the model to predict flatness. Must be given path argument.
    :param X_test: 4d numpy array with bacterial images
    :return prediction: float probability of flatness
    """

    # TODO: Refactor magic variable
    flat_path = 'neural_network/models/'

    model = load_model(flat_path)
    prediction = model.predict(X_test)

    return prediction

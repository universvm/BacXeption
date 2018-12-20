from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model


def convolve(input_layer, conv_depth, kernel_size, pool_size):
    """ Perform 3 convolutions + down pool"""

    conv_1 = Conv2D(conv_depth, (kernel_size, kernel_size),
                    padding='same', activation='relu')(input_layer)
    conv_2 = Conv2D(conv_depth, (kernel_size, kernel_size),
                    padding='same', activation='relu')(conv_1)
    conv_3 = Conv2D(conv_depth, (kernel_size, kernel_size),
                    padding='same', activation='relu')(conv_2)

    return MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)


def build_network():
    """ Main CNN to predict flat or not_flat  """

    # Hyperparameters:
    conv_depth = [32, 64, 128]
    kernel_size = 3
    pool_size = 3
    hidden_size = 50

    # Network structure:
    inp_layer = Input(shape=(180, 180, 1))

    conv_layer = inp_layer
    for i in range(3):
        conv_layer = convolve(conv_layer, conv_depth[i], kernel_size,
                              pool_size)
    flat = Flatten()(conv_layer)

    hidden = Dense(hidden_size, activation='relu')(flat)
    softmax = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=inp_layer, outputs=softmax)

    model.compile(loss='mean_squared_error', optimizer='adamax',
                  metrics=['accuracy'])

    # Output summary:
    print(model.summary())

    return model


from tensorflow import keras
from tensorflow.keras import regularizers


def define_model():
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',
                                  padding='valid', strides=(1, 1), input_shape=None))
    model.add(keras.layers.MaxPooling2D(
        (2, 2), strides=(1, 1), padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (2, 2), strides=(
        1, 1), padding='valid', activation='relu'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                 bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                 bias_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5)))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    MaxPooling2D,
    Conv2D,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    GlobalAvgPool2D,
)
from tensorflow.keras.models import Sequential

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            BatchNormalization(),
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def resnet34(x, y, scale, iter=150):
    print("setting gpu options")
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print("seting input to list")
    try:
        x.shape
        x = x.tolist()
    except:
        pass
    print("creating tensor")
    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    dim_persist = int(np.shape(x)[1] ** 0.5)
    x = x.reshape((np.shape(x)[0], dim_persist, dim_persist))
    x = np.expand_dims(x, -1)
    print(np.shape(x))
    samples = int(np.shape(x)[0])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    print("Input vector size: " + str(input_dim))
    model = Sequential()
    model.add(
        Conv2D(
            filters=64,
            kernel_size=5,
            activation="relu",
            input_shape=(dim_persist, dim_persist, 1),
            strides=1,
            data_format="channels_last",
            use_bias=False,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
    prev_filters = 64
    for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(GlobalAvgPool2D())
    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Dense(512))

    model.add(Dense(1, activation="linear"))
    model.summary()

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(optimizer=opt, loss="MSE", metrics=["MeanSquaredError", "MAE"])
    early_stop = EarlyStopping(monitor="loss", verbose=1, patience=10)
    history = model.fit(
        x_train,
        y_train,
        epochs=iter,
        batch_size=32,
        callbacks=[early_stop],
        validation_split=0.15,
    )
    ret = model.evaluate(x_test, y_test, verbose=2)
    plt.plot(history.history["loss"][2:-1], label="Training Loss")
    plt.plot(history.history["val_loss"][2:-1], label="Validation Loss")
    plt.legend()
    score = str(mean_squared_error(model.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(model.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(model.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    score_mae = mean_absolute_error(model.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return model




class AlexNet(tf.keras.Model):
    """
    AlexNet used for regression task
    """
    def __init__(self, input_shape=(224, 224, 3)):
        super(AlexNet, self).__init__()
        # 1st conv layer                 # original: (4,4)
        self.conv_1 = Conv2D(96, (11,11), strides=(2,2), activation='relu', input_shape=input_shape, padding='same')
        self.bn_1 = BatchNormalization()
        self.max_1 = MaxPooling2D((3,3), strides=(2,2)) # overlapping pooling described in the paper

        # 2nd conv layer
        self.conv_2 = Conv2D(256, (5,5), activation='relu', padding='same')
        self.bn_2 = BatchNormalization()
        self.max_2 = MaxPooling2D((3,3), strides=(2,2))

        # 3rd conv layer
        self.conv_3 = Conv2D(384, (3,3), activation='relu', padding='same')

        # 4th conv layer
        self.conv_4 = Conv2D(384, (3,3), activation='relu', padding='same')

        # 5th conv layer
        self.conv_5 = Conv2D(256, (3,3), activation='relu', padding='same')
        self.max_5 = MaxPooling2D((3,3), strides=(2,2))

        # Flatten
        self.flatten = Flatten()

        # 1st fc layer
        self.fc_1 = Dense(2048, activation='relu') # original:4096
        self.drop_1 = Dropout(0.5)

        # 2nd fc_layer
        self.fc_2 = Dense(2048, activation='relu') # original:4096
        self.drop_2 = Dropout(0.5)

        # 3rd fc_layer (output layer)
        self.fc_3 = Dense(1, activation='linear')

    def call(self, inputs):
        """
        model's forward pass
        """
        # 1st conv
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.max_1(x)

        # 2nd conv
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.max_2(x)

        # 3rd conv
        x = self.conv_3(x)

        # 4th conv
        x = self.conv_4(x)

        # 5th conv
        x = self.conv_5(x)
        x = self.max_5(x)

        x = self.flatten(x)

        # 1st fc
        x = self.fc_1(x)
        x = self.drop_1(x)

        # 2nd fc
        x = self.fc_2(x)
        x = self.drop_2(x)

        # 3rd fc: output layer (regression task)
        x = self.fc_3(x)

        return x



def alexnet(x, y, scale, iter=150):
    print("setting gpu options")
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print("seting input to list")
    try:
        x.shape
        x = x.tolist()
    except:
        pass
    print("creating tensor")
    try:
        x = tf.convert_to_tensor(x.tolist())
        y = tf.convert_to_tensor(y.tolist())
        input_dim = np.shape(x[0])

    except:
        input_dim = len(x[0])

    x = np.array(x)
    y = np.array(y)

    dim_persist = int(np.shape(x)[1] ** 0.5)
    x = x.reshape((np.shape(x)[0], dim_persist, dim_persist))
    x = np.expand_dims(x, -1)
    print(np.shape(x))
    samples = int(np.shape(x)[0])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    print("Input vector size: " + str(input_dim))
    # instantiate model
    model = AlexNet(input_shape=(x_train.shape[1:]))
    model.build(input_shape=x_train.shape)
    model.summary()

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(optimizer=opt, loss="MSE", metrics=["MeanSquaredError", "MAE"])
    early_stop = EarlyStopping(monitor="loss", verbose=1, patience=10)
    history = model.fit(
        x_train,
        y_train,
        epochs=iter,
        batch_size=32,
        callbacks=[early_stop],
        validation_split=0.15,
    )
    ret = model.evaluate(x_test, y_test, verbose=2)
    plt.plot(history.history["loss"][2:-1], label="Training Loss")
    plt.plot(history.history["val_loss"][2:-1], label="Validation Loss")
    plt.legend()
    score = str(mean_squared_error(model.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(model.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(model.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    score_mae = mean_absolute_error(model.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return model

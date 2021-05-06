import numpy as np
import tensorflow as tf

from config import REG_CONST, HIDDEN_CNN_LAYERS, LEARNING_RATE, MOMENTUM

class ResidualCNN:

    def predict(self, input):
        return self.model.predict(input)

    def read(self, path):
        self.model.set_weights(tf.keras.models.load_model(path).get_weights())

    def write(self, path):
        self.model.save(path)

    def __addConvLayer(self, layers, filters, kernelSize):
        layers = tf.keras.layers.Conv2D(
            filters, 
            kernelSize, 
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST)
        )(layers)
        layers = tf.keras.layers.BatchNormalization(axis=1)(layers)
        layers = tf.keras.layers.LeakyReLU()(layers)
        return layers

    def __addResidualLayer(self, layers, filters, kernelSize):
        x = self.__addConvLayer(layers, filters, kernelSize)
        x = tf.keras.layers.Conv2D(
            filters,
            kernelSize,
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST)
        )(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.add([layers, x])
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def __getValueHead(self, layers):
        layers = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1,1),
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST)
        )(layers)
        layers = tf.keras.layers.BatchNormalization(axis=1)(layers)
        layers = tf.keras.layers.LeakyReLU()(layers)
        layers = tf.keras.layers.Flatten()(layers)
        layers = tf.keras.layers.Dense(
            20,
            use_bias=False,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST)
        )(layers)
        layers = tf.keras.layers.LeakyReLU()(layers)
        layers = tf.keras.layers.Dense(
            1,
            use_bias=False,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST),
            name="value_head"
        )(layers)
        return layers

    def __getPolicyHead(self, layers):
        layers = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(1,1),
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST)
        )(layers)
        layers = tf.keras.layers.BatchNormalization(axis=1)(layers)
        layers = tf.keras.layers.LeakyReLU()(layers)
        layers = tf.keras.layers.Flatten()(layers)
        layers = tf.keras.layers.Dense(
            384,
            use_bias=False,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(REG_CONST),
            name="policy_head"
        )(layers)
        return layers

    def __init__(self):
        self.model = self.__buildModel()

    def __buildModel(self):
        inputLayer = tf.keras.layers.Input(shape = (11, 6, 4), name="input_layer")
        hiddenLayers = self.__addConvLayer(inputLayer, HIDDEN_CNN_LAYERS[0]["filters"], HIDDEN_CNN_LAYERS[0]["kernel_size"])
        for layerConfig in HIDDEN_CNN_LAYERS[1:]:
            hiddenLayers = self.__addResidualLayer(hiddenLayers, layerConfig["filters"], layerConfig["kernel_size"])
        valueHead = self.__getValueHead(hiddenLayers)
        policyHead = self.__getPolicyHead(hiddenLayers)

        model = tf.keras.models.Model(inputs=[inputLayer], outputs=[valueHead, policyHead])
        model.compile(
            optimizer=tf.keras.optimizers.SGD(LEARNING_RATE, MOMENTUM),
            loss={
                "value_head": "mean_squared_error",
                "policy_head": "categorical_crossentropy"
            }
        )
        return model

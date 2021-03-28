import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers


class AgeModel(tf.keras.Model):
    def __init__(self, model_metadata, data_format):
        super(AgeModel, self).__init__()
        self.data_format = data_format
        self.model_backbone = self.construct_model(model_metadata[:-1])
        assert model_metadata[-1]["type"] == "dense"
        self.classify_layer = self.dense_layer(model_metadata[-1])
        self.model_metadata = model_metadata # Holds the information about layer construction
        print("INFO: Model constructed")


    def call(self, inputs):
        x = self.model_backbone(inputs)
        return self.classify_layer(x)


    def construct_model(self, model_metadata):
        model = tf.keras.Sequential()
        for layer_metadata in model_metadata:
            if "type" not in layer_metadata.keys():
                raise RuntimeError("Invalid layer entry, layer has no type in {}".format(layer_metadata))
            if layer_metadata["type"] == "conv2d":
                model.add(self.conv2d_layer(layer_metadata))
            elif layer_metadata["type"] == "dense":
                model.add(self.dense_layer(layer_metadata))
            elif layer_metadata["type"] == "pool":
                model.add(self.pool_layer(layer_metadata))
            elif layer_metadata["type"] == "batch_norm":
                model.add(self.batchnorm_layer(layer_metadata))
            elif layer_metadata["type"] == "flatten":
                model.add(layers.Flatten(data_format=self.data_format))
            else:
                raise RuntimeError("Invalid layer type {}, should be one of (conv2d, dense, pool, batch_norm)")
        return model


    def conv2d_layer(self, layer_metadata):
        if layer_metadata["initializer"] == "xavier":
            initializer = initializers.GlorotNormal()
        elif layer_metadata["initializer"] == "random":
            initializer = initializers.RandomNormal()
        else:
            raise ValueError("Specified initializer for {} is invalid: should be one of (xavier, random)".format(layer_metadata))

        if layer_metadata["regularizer"] not in ("l1", "l2", None):
            raise ValueError("Specified regularizer for {} is invalid: should be one of (l1, l2, None)".format(layer_metadata))
        else:
            regularizer = layer_metadata["regularizer"]

        if layer_metadata["activation"] not in ("relu", "sigmoid, softmax", "tanh"):
            raise ValueError("Activation specified for {} is invalid, should be one of (relu, sigmoid, softmax, tanh)")

        return layers.Conv2D(filters=layer_metadata["filters"],kernel_size=layer_metadata["kernel_size"], strides=layer_metadata["strides"],
            padding=layer_metadata["padding"],data_format=self.data_format, activation=layer_metadata["activation"], kernel_initializer=initializer, 
            bias_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)


    def dense_layer(self, layer_metadata):
        if layer_metadata["activation"] not in ("relu", "sigmoid, softmax", "tanh"):
            raise ValueError("Activation specified for {} is invalid, should be one of (relu, sigmoid, softmax, tanh)")

        if layer_metadata["initializer"] == "xavier":
            initializer = initializers.GlorotNormal()
        elif layer_metadata["initializer"] == "random":
            initializer = initializers.RandomNormal()
        else:
            raise ValueError("Specified initializer for {} is invalid: should be one of (xavier, random)".format(layer_metadata))

        if layer_metadata["regularizer"] not in ("l1", "l2", None):
            raise ValueError("Specified regularizer for {} is invalid: should be one of (l1, l2, None)".format(layer_metadata))
        else:
            regularizer = layer_metadata["regularizer"]

        return layers.Dense(units=layer_metadata["units"], activation=layer_metadata["activation"], kernel_initializer=initializer, bias_initializer=initializer, 
            kernel_regularizer=regularizer, bias_regularizer=regularizer)


    def pool_layer(self, layer_metadata):
        if layer_metadata["pool_type"] == "avg":
            return layers.AvgPool2D(pool_size=layer_metadata["pool_size"], strides=layer_metadata["strides"], padding=layer_metadata["padding"], data_format=self.data_format)
        elif layer_metadata["pool_type"] == "max":
            return layers.MaxPool2D(pool_size=layer_metadata["pool_size"], strides=layer_metadata["strides"], padding=layer_metadata["padding"], data_format=self.data_format)
        else:
            raise ValueError("Invalid pooling layer type for {}, should be one of (avg, max)".format(layer_metadata))


    def batchnorm_layer(self, layer_metadata):
        if "momentum" in layer_metadata.keys():
            momentum = layer_metadata["momentum"]
        else:
            momentum = 0.99
        
        if "epsilon" in layer_metadata.keys():
            epsilon = layer_metadata["epsilon"]
        else:
            epsilon = 0.001
        
        return layers.BatchNormalization(momentum=momentum, epsilon=epsilon)



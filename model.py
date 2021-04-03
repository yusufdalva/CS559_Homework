import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers

# TODO: Add batch_norm option to the layers


class AgeModel(tf.keras.Model):
    def __init__(self, model_metadata, data_format):
        super(AgeModel, self).__init__()
        self.data_format = data_format
        self.net_layers = []
        self.construct_model(model_metadata)
        self.model_metadata = model_metadata # Holds the information about layer construction
        assert model_metadata[-1]["type"] == "dense" and model_metadata[-1]["units"] == 1


    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def build_comp_graph(self, input_shape):
        instance_shape = input_shape[1:]
        self.build(input_shape)
        inputs = layers.Input(shape=instance_shape)
        self.call(inputs)


    def construct_model(self, model_metadata):
        for layer_metadata in model_metadata:
            if "type" not in layer_metadata.keys():
                raise RuntimeError("Invalid layer entry, layer has no type in {}".format(layer_metadata))
            if layer_metadata["type"] == "conv2d":
                self.net_layers.append(self.conv2d_layer(layer_metadata))
                if "batch_norm" in layer_metadata.keys() and layer_metadata["batch_norm"] == True:
                    self.net_layers.append(self.batchnorm_layer(layer_metadata))
                    self.net_layers.append(tf.keras.layers.Activation(layer_metadata["activation"]))
            elif layer_metadata["type"] == "dense":
                self.net_layers.append(self.dense_layer(layer_metadata))
                if "batch_norm" in layer_metadata.keys() and layer_metadata["batch_norm"] == True:
                    self.net_layers.append(self.batchnorm_layer(layer_metadata))
                    self.net_layers.append(tf.keras.layers.Activation(layer_metadata["activation"]))
            elif layer_metadata["type"] == "pool":
                self.net_layers.append(self.pool_layer(layer_metadata))
            elif layer_metadata["type"] == "batch_norm":
                self.net_layers.append(self.batchnorm_layer(layer_metadata))
            elif layer_metadata["type"] == "flatten":
                self.net_layers.append(layers.Flatten(data_format=self.data_format))
            elif layer_metadata["type"] == "dropout":
                self.net_layers.append(self.dropout_layer(layer_metadata))
            else:
                raise RuntimeError("Invalid layer type {}, should be one of (conv2d, dense, pool, batch_norm, dropout)")


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
            regularizer = self.get_regularizer(layer_metadata["regularizer"], layer_metadata["reg_ratio"])

        if layer_metadata["activation"] not in ("relu", "sigmoid, softmax", "tanh"):
            raise ValueError("Activation specified for {} is invalid, should be one of (relu, sigmoid, softmax, tanh)")

        if "batch_norm" in layer_metadata.keys() and layer_metadata["batch_norm"] == True:
            return layers.Conv2D(filters=layer_metadata["filters"],kernel_size=layer_metadata["kernel_size"], strides=layer_metadata["strides"],
            padding=layer_metadata["padding"],data_format=self.data_format, activation=None, kernel_initializer=initializer, 
            bias_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)

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
            regularizer = self.get_regularizer(layer_metadata["regularizer"], layer_metadata["reg_ratio"])


        if layer_metadata["activation"] not in ("relu", "sigmoid, softmax", "tanh"):
            raise ValueError("Activation specified for {} is invalid, should be one of (relu, sigmoid, softmax, tanh)")

        if "batch_norm" in layer_metadata.keys() and layer_metadata["batch_norm"] == True:
            return layers.Dense(units=layer_metadata["units"], activation=None, kernel_initializer=initializer, bias_initializer=initializer, 
            kernel_regularizer=regularizer, bias_regularizer=regularizer)

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


    def dropout_layer(self, layer_metadata):
        # Default dropout rate is set as 0.5
        if "rate" in layer_metadata.keys():
            rate = layer_metadata["rate"]
        else:
            rate = 0.5
        return layers.Dropout(rate)

    def get_regularizer(self, type, ratio=0.01):
        if type == "l1":
            return tf.keras.regularizers.l1(l1=ratio)
        elif type == "l2":
            return tf.keras.regularizers.l2(l2=ratio)
        return None


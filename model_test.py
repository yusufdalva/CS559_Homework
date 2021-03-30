import tensorflow as tf
import tensorflow.keras.losses as losses
from model import AgeModel
import os
from data.data_utils import Dataset_numpy

if __name__ == "__main__":
    print("Tensorflow version used: {}".format(tf.__version__))
    data_path = os.path.join(os.getcwd(), 'data')
    data_path = os.path.join(data_path, 'UTKFace_downsampled')
    train_path = os.path.join(data_path, 'training_set')
    val_path = os.path.join(data_path, 'validation_set')
    test_path = os.path.join(data_path, 'test_set')
    dataset = Dataset_numpy(train_path, val_path, test_path, 'jpg')
    # Monitor dataset details
    ## Training set
    print("Training data shape: {}".format(dataset.train_data[0].shape))
    print("Training labels shape: {}".format(dataset.train_data[1].shape))
    ## Validation set
    print("Validation data shape: {}".format(dataset.val_data[0].shape))
    print("Validation labels shape: {}".format(dataset.val_data[1].shape))
    ## Test set
    print("Testing data shape: {}".format(dataset.test_data[0].shape))
    print("Testing labels shape: {}".format(dataset.test_data[1].shape))

    # Model construction
    model_metadata = []
    model_metadata.append({"type": "conv2d", "filters": 32, "kernel_size": (5,5), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "conv2d", "filters": 32, "kernel_size": (5,5), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "batch_norm"})
    model_metadata.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
    model_metadata.append({"type": "conv2d", "filters": 64, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "conv2d", "filters": 64, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "batch_norm"})
    model_metadata.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
    model_metadata.append({"type": "conv2d", "filters": 128, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "conv2d", "filters": 128, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "batch_norm"})
    model_metadata.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
    model_metadata.append({"type": "flatten"})
    model_metadata.append({"type": "dense", "units": 128, "activation": "relu", "initializer": "xavier", "regularizer": None})
    model_metadata.append({"type": "dense", "units": 1, "activation": "relu", "initializer": "xavier", "regularizer": None})
    
    model = AgeModel(model_metadata, "channels_last")
    model.build_comp_graph((None, 91, 91, 1)) # Building computational graph to monitor dimensions of layer matrices
    model.summary()

    # Configure model training with optimizer and loss
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(loss=loss, optimizer=optim)

    # Fit training data to model, with validation set performance
    print("TRAINING")
    model.fit(x=dataset.train_data[0], y=dataset.train_data[1], epochs=10, verbose=1, batch_size=32, validation_data=dataset.val_data)
    print("TRAINING DONE")

    # Evaluate model on test set
    print("EVALUATION")
    model.evaluate(x=dataset.test_data[0], y=dataset.test_data[1])
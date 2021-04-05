import tensorflow as tf
from data.data_utils import Dataset, save_dataset
import os
from model import AgeModel
import tensorflow.keras.losses as losses
import matplotlib.pyplot as plt
import h5py
import numpy as np
from time import time
from data.data_utils import Dataset, save_dataset

# Fix the random seed
seed_value=7777
tf.compat.v1.set_random_seed(seed_value)
np.random.seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

# Set following config to resolve GPU errors
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
print("Tensorflow version used: {}".format(tf.__version__))

from tensorflow.python.client import device_lib
def check_if_gpu_used():
    gpu_names = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if len(gpu_names) >= 1:
        print("Number of GPUs used by Tensorflow: {}".format(len(gpu_names)))
    else:
        print("Tensorflow operates on CPU now.")
check_if_gpu_used()

data_path = os.path.join(os.getcwd(), 'data')
datafile_path = os.path.join(data_path, "dataset.h5")
#save_dataset(data_path, dataset) # Compressing data in h5 format
start = time()
f = h5py.File(datafile_path, "r")
train_samples = np.array(f["train_samples"])
train_labels = np.array(f["train_labels"])
val_samples = np.array(f["val_samples"])
val_labels = np.array(f["val_labels"])
test_samples = np.array(f["test_samples"])
test_labels = np.array(f["test_labels"])
end = time()
f.close()
print("Monitoring compressed data details")
## Training set
print("Training data shape: {}".format(train_samples.shape))
print("Training labels shape: {}".format(train_labels.shape))
## Validation set
print("Validation data shape: {}".format(val_samples.shape))
print("Validation labels shape: {}".format(val_labels.shape))
## Test set
print("Testing data shape: {}".format(test_samples.shape))
print("Testing labels shape: {}".format(test_labels.shape))
print("Time to construct the dataset from compressed file: {:.3f} seconds".format(end - start))

# Model specification
non_regularized_model = []
non_regularized_model.append({"type": "conv2d", "filters": 32, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "conv2d", "filters": 32, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None,"reg_ratio": None})
non_regularized_model.append({"type": "batch_norm"})
non_regularized_model.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
non_regularized_model.append({"type": "conv2d", "filters": 64, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "conv2d", "filters": 64, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "batch_norm"})
non_regularized_model.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
non_regularized_model.append({"type": "conv2d", "filters": 128, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "conv2d", "filters": 128, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "conv2d", "filters": 128, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "batch_norm"})
non_regularized_model.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
non_regularized_model.append({"type": "conv2d", "filters": 256, "kernel_size": (3,3), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "conv2d", "filters": 512, "kernel_size": (1,1), "strides":(1,1), "padding":"valid", "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "batch_norm"})
non_regularized_model.append({"type": "pool", "pool_type": "max", "pool_size": (2,2), "strides": None, "padding": "valid"})
non_regularized_model.append({"type": "flatten"})
non_regularized_model.append({"type": "dense", "units": 256, "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "dense", "units": 128, "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "dense", "units": 64, "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})
non_regularized_model.append({"type": "dense", "units": 1, "activation": "relu", "initializer": "xavier", "regularizer": None, "reg_ratio": None})

def get_model_with_reg(template_model, reg_ratio):
    reg_model = template_model.copy()
    for layer in template_model:
        if layer["type"] in ("dense", "conv2d"):
            layer["regularizer"] = "l2"
            layer["reg_ratio"] = reg_ratio
    return reg_model

from tqdm import tqdm
def optimize_lr_reg(step_count, template_model, training_samples, training_labels, valid_samples, valid_labels):
    lr_exp = np.random.uniform(-6, -3, step_count)
    reg_exp = np.random.uniform(-4, 1, step_count)
    results = []
    # Results stored as (val_loss, train_loss, lr, reg_ratio)
    for exp_idx in tqdm(range(step_count)):
        model_metadata = get_model_with_reg(template_model, 10**reg_exp[exp_idx])
        exp_model = AgeModel(model_metadata, "channels_last")
        exp_model.build_comp_graph((None, 91, 91, 1))
        optim = tf.keras.optimizers.Adam(learning_rate=10**lr_exp[exp_idx])
        loss = tf.keras.losses.MeanAbsoluteError()
        exp_model.compile(loss=loss, optimizer=optim)
        history = exp_model.fit(x=training_samples, y=training_labels, epochs=5, verbose=0, validation_data=(valid_samples, valid_labels))
        results.append((history.history["val_loss"][-1], history.history["loss"][-1], lr_exp[exp_idx], reg_exp[exp_idx]))
    results.sort(key=lambda x: x[0])
    return results
#exp_results = optimize_lr_reg(100, non_regularized_model, train_samples[random_idx[:500]], train_labels[random_idx[:500]], val_samples, val_labels)
# Best 10 results - by validation loss
#print(exp_results[:10])

def lr_reg_small_range(step_count, template_model, training_samples, training_labels, valid_samples, valid_labels):
    lr_exp = np.random.uniform(-4.5, -2.5, step_count)
    reg_exp = np.random.uniform(-4.5, -2.5, step_count)
    results = []
    # Results stored as (val_loss, train_loss, lr, reg_ratio)
    for exp_idx in tqdm(range(step_count)):
        model_metadata = get_model_with_reg(template_model, 10**reg_exp[exp_idx])
        exp_model = AgeModel(model_metadata, "channels_last")
        exp_model.build_comp_graph((None, 91, 91, 1))
        optim = tf.keras.optimizers.Adam(learning_rate=10**lr_exp[exp_idx])
        loss = tf.keras.losses.MeanAbsoluteError()
        exp_model.compile(loss=loss, optimizer=optim)
        history = exp_model.fit(x=training_samples, y=training_labels, epochs=5, verbose=0, validation_data=(valid_samples, valid_labels))
        results.append((history.history["val_loss"][-1], history.history["loss"][-1], lr_exp[exp_idx], reg_exp[exp_idx]))
    results.sort(key=lambda x: x[0])
    return results
#exp_results_2 = lr_reg_small_range(20, non_regularized_model, train_samples[random_idx[:500]], train_labels[random_idx[:500]], val_samples, val_labels)
# Best 10 results - by validation loss
#print(exp_results_2[:10])

def visualize_experiment(history):
    losses = history.history["loss"]
    valid_losses = history.history["val_loss"]
    print("Validation loss for experiment: {}".format(losses[-1]))
    print("Minimum validation loss value achieved: {}".format(np.amin(valid_losses)))
    plt.plot(range(1, len(losses) + 1), losses, color="blue", label="Training Loss")
    plt.plot(range(1, len(losses) + 1), valid_losses, color="orange", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss plot for the regularized model")
    plt.legend(loc="upper left")
    plt.show()


def batchnorm_opt_1(template_model, reg_ratio, lr, training_samples, training_labels, valid_samples, valid_labels):
    batchnorm_model = template_model.copy()
    batchnorm_model = get_model_with_reg(template_model, reg_ratio)
    exp_model = AgeModel(batchnorm_model, "channels_last")
    exp_model.build_comp_graph((None, 91, 91, 1))
    exp_model.summary()
    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanAbsoluteError()
    exp_model.compile(loss=loss, optimizer=optim)
    history = exp_model.fit(x=training_samples, y=training_labels, epochs=1000, verbose=1, validation_data=(valid_samples, valid_labels))
    return history, exp_model

hist, t_model = batchnorm_opt_1(non_regularized_model, 1e-5, 1e-4, train_samples, train_labels, val_samples, val_labels)
t_model.save_weights(os.path.join(os.getcwd(), 'model/'),save_format='tf')
visualize_experiment(hist)
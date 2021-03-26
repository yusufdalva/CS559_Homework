from zipfile import ZipFile
import os
import tensorflow as tf
import tensorflow.keras.preprocessing.image as tf_img
import numpy as np

def unzip_dataset(dataset_path, path_to_extract):
    if not os.path.isdir(path_to_extract):
        os.mkdir(path_to_extract)
    if not os.path.isfile(dataset_path):
        raise ValueError('There is no file such as {}'.format(dataset_path))
    with ZipFile(dataset_path, 'r') as dataset_ref:
        dataset_ref.extractall(path_to_extract)


class Dataset():
    """ Constructor for the dataset class, the inputs are:
    - img_size: image dimensions in (width, height, channels)
    - train_path: path to training set
    - val_path: path to validation set
    - test_path: path to test set 
    - data_format: jpg or png """
    def __init__(self, batch_size, train_path, val_path, test_path, data_format, color_mode='grayscale'):
        assert os.path.isdir(train_path)
        assert os.path.isdir(val_path)
        assert os.path.isdir(test_path)
        assert data_format in ('jpg', 'png')
        assert color_mode in ('grayscale', 'rgb')
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.train_data = self.process_data_path(train_path, data_format)
        self.val_data = self.process_data_path(val_path, data_format)
        self.test_data = self.process_data_path(test_path, data_format)
        print('INFO: Dataset constructed')

    def process_data_path(self, data_path, data_format):
        if data_path[-1] != '/':
            data_path += '/'
        if data_format == 'jpg':
            file_names = tf.data.Dataset.list_files(data_path + '*.jpg')
        elif data_format == 'png':
            file_names = tf.data.Dataset.list_files(data_path + '*.png')
        data = file_names.map(self.process_image)
        return data

    def process_image(self, img_path):
        image = tf.io.read_file(img_path)
        if self.color_mode == 'grayscale':
            image = tf.image.decode_jpeg(image, channels=1) / 255
        else:
            image = tf.image.decode_jpeg(image, channels=3) / 255
        filename = tf.strings.split(img_path, sep='/')[-1]
        label = tf.strings.to_number(tf.strings.split(filename, '_')[0])
        return image, label


if __name__ == '__main__':
    print(tf.__version__)
    data_path = os.path.join(os.getcwd(), 'data')
    data_path = os.path.join(data_path, 'UTKFace_downsampled')
    train_path = os.path.join(data_path, 'training_set')
    val_path = os.path.join(data_path, 'validation_set')
    test_path = os.path.join(data_path, 'test_set')
    dataset = Dataset(32, train_path, val_path, test_path, 'jpg')

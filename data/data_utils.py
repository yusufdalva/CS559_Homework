from zipfile import ZipFile
import os

def unzip_dataset(dataset_path, path_to_extract):
    if not os.path.isdir(path_to_extract):
        os.mkdir(path_to_extract)
    if not os.path.isfile(dataset_path):
        raise ValueError('There is no file such as {}'.format(dataset_path))
    with ZipFile(dataset_path, 'r') as dataset_ref:
        dataset_ref.extractall(path_to_extract)


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data')
    file_path = os.path.join(data_path, 'UTKFace_downsampled.zip')
    unzip_dataset(file_path, data_path)
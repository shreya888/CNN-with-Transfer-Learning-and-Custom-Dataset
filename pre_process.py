from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.tensor
from sklearn.model_selection import train_test_split
import glob


def data_loaders(batch_size):
    """
    Function to load, read, split, transform image dataset into required tensors
    :param batch_size: size of each batch for data loader
    :return: train, val and test data loaders
    """
    # Get list of files in train and test directories
    BASE_DIR   = "./engn8536/Dataset/"
    TRAIN_DIR  = os.path.join(BASE_DIR, 'cat-dog-train/')
    TEST_DIR   = os.path.join(BASE_DIR, 'cat-dog-test/')
    train_list = glob.glob(os.path.join(TRAIN_DIR, '*.tif'))
    test_list  = glob.glob(os.path.join(TEST_DIR, '*.tif'))

    # Split train_files into train and validation sets
    train_list, val_list = train_test_split(train_list, test_size=0.1)

    def get_data_loaders(data_list, train=False):
        """
        Helper function to get customized data loaders from list of data paths
        :param train: if True, then transform data as for training else for val or test
        :param data_list: list of paths of image data
        :return:
        """
        # Get labels
        data_labels = label_data(data_list)
        # Apply transforms on images and get tensors
        if train:
            data_list = train_image_transform(data_list)
        else:
            data_list = image_transform_all(data_list)
        # Pair image tensors with their label in a tensor dataset
        data = TensorDataset(torch.stack(data_list), torch.tensor(data_labels))
        # Shuffle the data and get data loaders of specified batch_size
        data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
        return data_loader

    train = get_data_loaders(train_list, True)
    val   = get_data_loaders(val_list, False)
    test  = get_data_loaders(test_list, False)

    print('Loaded images into custom data loaders...')

    return train, val, test


def label_data(files_path):
    """
    Generates a list of labels according to filenames - 0 denotes label 'cat' and 1 denotes label 'dog'
    :param files_path: path of image files folder
    :return: list of labels
    """
    labels = []
    for filename in files_path:
        label = filename.split('/')[4].split('.')[0]
        if 'cat' == label:
            labels.append(0.0)
        elif 'dog' == label:
            labels.append(1.0)
    return labels


def image_transform_all(dir_path):
    """
    Function to resize image and normalize data when loading data
    :param dir_path: directory path of images
    :return: transformed image tensor
    """
    image_tensor = []
    for image_path in dir_path:
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
        image_tensor.append(transform(image))
    return image_tensor


def train_image_transform(dir_path):
    """
    Perform data augmentation and transform when training - random flip, padding, crop images
    :param dir_path: directory path of images
    :return: transformed image tensor
    """
    image_tensor = []
    for image_path in dir_path:
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(4),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])
        transformed_image = transform(image)
        image_tensor.append(transformed_image)
    return image_tensor



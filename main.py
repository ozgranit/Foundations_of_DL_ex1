import random
import pickle

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def unpickle(file):
    with open(file, 'rb') as fo:
        pic_dict = pickle.load(fo, encoding='bytes')

    return pic_dict


def sample_from_dict(pic_dict: dict, sample_size: float=0.1) -> dict:

    keys = pic_dict.keys()
    num_samples = int(len(keys) * sample_size)

    rand_keys = random.sample(keys, num_samples)
    sample_pic_dict = {k: pic_dict[k] for k in rand_keys}

    return sample_pic_dict


def load_train_data(cifar_folder_path: Path) -> dict:
    train_data = {}

    for i in range(1, 6):
        file = cifar_folder_path / f"data_batch_{i}"
        train_data.update(unpickle(file))
        print(f"loaded {file}")

    return train_data


def load_test_data(cifar_folder_path: Path) -> dict:

    file = cifar_folder_path / "test_batch"
    test_data = unpickle(file)
    print(f"loaded {file}")

    return test_data


def preprocess_data(data):
    images = data[b'data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    labels = np.array(data[b'labels'])

    return images, labels


def get_data(cifar_folder_path: Path):

    # Load the training and test data
    train_data = load_train_data(cifar_folder_path)
    test_data = load_test_data(cifar_folder_path)

    # Preprocess the data
    train_images, train_labels = preprocess_data(train_data)
    test_images, test_labels = preprocess_data(test_data)

    # Subsample 10% of the original data
    sample_size = 0.1
    num_train_samples = int(len(train_images) * sample_size)
    num_test_samples = int(len(test_images) * sample_size)

    train_images = train_images[:num_train_samples]
    train_labels = train_labels[:num_train_samples]
    test_images = test_images[:num_test_samples]
    test_labels = test_labels[:num_test_samples]

    # Normalize the pixel values of the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def svm_pred(svm, train_images, train_labels, test_images, test_labels):

    svm.fit(train_images.reshape(len(train_images), -1), train_labels)
    linear_train_preds = svm.predict(train_images.reshape(len(train_images), -1))
    linear_test_preds = svm.predict(test_images.reshape(len(test_images), -1))

    train_acc = accuracy_score(train_labels, linear_train_preds)
    test_acc = accuracy_score(test_labels, linear_test_preds)

    return train_acc, test_acc


def baseline(train_images, train_labels, test_images, test_labels):

    # Train a linear SVM
    linear_svm = SVC(kernel='linear')
    linear_train_acc, linear_test_acc = svm_pred(linear_svm, train_images, train_labels, test_images, test_labels)

    print('Linear SVM:')
    print(f'Train accuracy: {linear_train_acc:.4f}')
    print(f'Test accuracy: {linear_test_acc:.4f}')

    # Train an RBF SVM
    rbf_svm = SVC(kernel='rbf')
    rbf_train_acc, rbf_test_acc = svm_pred(rbf_svm, train_images, train_labels, test_images, test_labels)

    print('RBF SVM:')
    print(f'Train accuracy: {rbf_train_acc:.4f}')
    print(f'Test accuracy: {rbf_test_acc:.4f}')


if __name__ == '__main__':
    cifar_folder_path = Path(r"C:\Users\ozgra\PycharmProjects\Foundations_of_DL\cifar-10-batches-py")
    # Part 1 - Setup and Baseline
    baseline(*get_data(cifar_folder_path))


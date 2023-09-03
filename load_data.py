import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def normalize(seq):
    return (seq - min(seq)) / (max(seq) - min(seq))

def remove_all_same(train_x, test_x):
    remove_idx = []
    for col in range(train_x.shape[1]):
        if max(train_x[:, col]) == min(train_x[:, col]):
            remove_idx.append(col)
        else:
            train_x[:, col] = normalize(train_x[:, col])

        if max(test_x[:, col]) == min(test_x[:, col]):
            remove_idx.append(col)
        else:
            test_x[:, col] = normalize(test_x[:, col])

    all_idx = set(range(train_x.shape[1]))
    remain_idx = list(all_idx - set(remove_idx))
    return train_x[:, remain_idx], test_x[:, remain_idx]

def slide_window(ts, window_size, stride):
    ts_length = ts.shape[0]
    samples = []
    for start in np.arange(0, ts_length, stride):
        if start + window_size > ts_length:
            break
        samples.append(ts[start : start + window_size])
    return np.array(samples)

def load_data(dataset_name, window_size, window_stride, batch_size):
    # root path
    root_path = os.path.join('dataset', dataset_name)

    # load data from .npy file
    train_data = np.load(os.path.join(root_path, 'train_data.npy'))
    test_data = np.load(os.path.join(root_path, 'test_data.npy'))

    # values and labels
    x_train, x_test = train_data[:, :-1], test_data[:, :-1]
    y_train, y_test = train_data[:, -1], test_data[:, -1]

    # remove columns have 0 variance
    x_train, x_test = remove_all_same(x_train, x_test)

    # number of features
    n_features = x_train.shape[1]

    # get window data
    x_train_slided = slide_window(x_train, window_size, window_stride)
    train_dataset = TensorDataset(torch.Tensor(x_train_slided), torch.Tensor(y_train[:len(x_train_slided)]))

    # get data loader
    data_loader = {"train": DataLoader(dataset = train_dataset,
                                       batch_size = batch_size,
                                       shuffle = False,
                                       drop_last = False),
                            "test": (x_test, y_test),
                            "n_features": n_features
    }

    return data_loader
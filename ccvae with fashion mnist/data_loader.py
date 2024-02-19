import errno
import os
import PIL
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

def split_fashionmnist(X, y, sup_frac, validation_num):
    """
    splits FashionMNIST
    """

    # validation set is the last 10,000 examples
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]

    X = X[0:-validation_num]
    y = y[0:-validation_num]

    if sup_frac == 0.0:
        return None, None, X, y, X_valid, y_valid

    if sup_frac == 1.0:
        return X, y, None, None, X_valid, y_valid

    split = int(sup_frac * len(X))
    X_sup = X[0:split]
    y_sup = y[0:split]
    X_unsup = X[split:]
    y_unsup = y[split:]

    return X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid

classes = 10

FashionMNIST_classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')

def custom_transform(x):
    return x.float() / 255

class FashionMNISTCached(FashionMNIST):
    # static class variables for caching training data
    classes = 10

    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    train_data, test_labels = None, None
    prior = torch.ones(1, classes) / classes
    fixed_imgs = None
    fixed_imgs_targets = None
    validation_size = 20000
    data_valid, labels_valid = None, None
    shape = (1, 64, 64)

    def prior_fn():
        return FashionMNISTCached.prior

    def clear_cache():
        FashionMNISTCached.train_data, FashionMNISTCached.test_labels = None, None

    def __init__(self, mode, sup_frac=None, *args, **kwargs):
        super(FashionMNISTCached, self).__init__(train=True if mode in ["sup", "unsup", "valid"] else 'test', *args, **kwargs)
        self.sub_label_inds = [i for i in range(classes)]
        self.mode = mode
        
        # self.transform = lambda x: (x/255).view(-1)
        self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.Lambda(custom_transform)
            ])

        assert mode in ["sup", "unsup", "test", "valid"], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if FashionMNISTCached.train_data is None:
                print("Splitting Dataset")

                FashionMNISTCached.train_data = self.data
                FashionMNISTCached.train_targets = self.targets

                FashionMNISTCached.train_data_sup, FashionMNISTCached.train_labels_sup, \
                    FashionMNISTCached.train_data_unsup, FashionMNISTCached.train_labels_unsup, \
                    FashionMNISTCached.data_valid, FashionMNISTCached.labels_valid = \
                    split_fashionmnist(FashionMNISTCached.train_data, FashionMNISTCached.train_targets,
                                 sup_frac, FashionMNISTCached.validation_size)

            if mode == "sup":
                self.data, self.targets = FashionMNISTCached.train_data_sup, FashionMNISTCached.train_labels_sup
                FashionMNISTCached.prior = torch.mean(torch.nn.functional.one_hot(self.targets).float(), dim=0)
            elif mode == "unsup":
                self.data = FashionMNISTCached.train_data_unsup
                # making sure that the unsupervised labels are not available to inference
                self.targets = FashionMNISTCached.train_labels_unsup * np.nan
            else:
                self.data, self.targets = FashionMNISTCached.data_valid, FashionMNISTCached.labels_valid

        else:
            self.data = self.data
            self.targets = self.targets

        # create a batch of fixed images
        if FashionMNISTCached.fixed_imgs is None:
            temp = []
            for i in range(64):
                temp.append(self.transform(self.data[i, None, :, :]))
            FashionMNISTCached.fixed_imgs = torch.stack(temp, dim=0)
                #temp.append([self.transform(self.data[i, None,:,:]),self.targets[i]])
            #FashionMNISTCached.fixed_imgs = temp
            
        if FashionMNISTCached.fixed_imgs_targets is None:
            temp = []
            for i in range(64):
                temp.append([self.transform(self.data[i, None,:,:]),self.targets[i]])
            FashionMNISTCached.fixed_imgs_targets = temp

    def __getitem__(self, index):
        
        X = self.transform(self.data[index, None,:,:])

        target = self.targets[index]
        
        return X, target

    def __len__(self):
        return len(self.data)
    

def setup_data_loaders(use_cuda, batch_size, sup_frac=1.0, root=None, cache_data=False, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_frac: fraction of supervised data examples
    :param cache_data: saves dataset to memory, prevents reading from file every time
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """

    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    cached_data = {}
    loaders = {}

    #clear previous cache
    FashionMNISTCached.clear_cache()

    if sup_frac == 0.0:
        modes = ["unsup", "test"]
    elif sup_frac == 1.0:
        modes = ["sup", "test", "valid"]
    else:
        modes = ["unsup", "test", "sup", "valid"]
        
    for mode in modes:
        cached_data[mode] = FashionMNISTCached(root=root, mode=mode, download=True, sup_frac=sup_frac)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
    return loaders


import os
import importlib
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


def prep_data_transforms(data_transform_flag):
    """ Get transform of dataset """

    data_transform_flag = data_transform_flag.lower()
    if data_transform_flag in ['cifar10', 'cifar100']:
        if data_transform_flag == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
        elif data_transform_flag == 'cifar100':
            mean = [0.5071, 0.4865, 0.4409]
            std = [0.2673, 0.2564, 0.2762]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transforms = None
    elif data_transform_flag in ['mnist', 'fashion_mnist']:
        if data_transform_flag == 'mnist':
            mean = [0.1307]
            std = [0.3081]
        elif data_transform_flag == 'fashion_mnist':
            mean = [0.2860]
            std = [0.3530]
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transforms = None
    else:
        try:
            transforms_lib = importlib.import_module(data_transform_flag)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    data_transform_flag))
    return train_transforms, val_transforms, test_transforms


def prep_dataset(dataset, dataset_dir, train_transforms, 
                 val_transforms, test_transforms, test_only):
    """ Get dataset for classification """

    if dataset == 'cifar10':
        train_set = datasets.CIFAR10(root=dataset_dir, train=True,
                                     download=True,
                                     transform=train_transforms)
        val_set = datasets.CIFAR10(root=dataset_dir, train=False,
                                   download=True, transform=val_transforms)
        test_set = None
    elif dataset == 'cifar100':
        train_set = datasets.CIFAR100(root=dataset_dir, train=True,
                                      download=True,
                                      transform=train_transforms)
        val_set = datasets.CIFAR100(root=dataset_dir, train=False,
                                    download=True, transform=val_transforms)
        test_set = None
    elif dataset == 'mnist':
        train_set = datasets.MNIST(root=dataset_dir, train=True,
                                   download=True, transform=train_transforms)
        val_set = datasets.MNIST(root=dataset_dir, train=False, 
                                 download=True, transform=val_transforms)
        test_set = None
    elif dataset == 'fashion_mnist':
        train_set = datasets.FashionMNIST(root=dataset_dir, train=True,
                                          download=True,
                                          transform=train_transforms)
        val_set = datasets.FashionMNIST(root=dataset_dir, train=False,
                                        download=True,
                                        transform=val_transforms)
        test_set = None
    else:
        try:
            dataset_lib = importlib.import_module(dataset)
            return dataset_lib.dataset(
                train_transforms, val_transforms, test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(dataset_dir))
    return train_set, val_set, test_set


def prep_data_loaders(train_set, val_set, test_set, dataset, data_loader_flag,
                      train_batch_size, val_batch_size, workers, drop_last,
                      test_only, train_sampler=None):
    """ Get data loader """

    if data_loader_flag in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']:
        train_loader = None
        if not test_only:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=train_batch_size,
                shuffle=(train_sampler is None),
                pin_memory=True, num_workers=workers,
                drop_last=drop_last, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=val_batch_size, shuffle=False,
            pin_memory=True, num_workers=workers,
            drop_last=drop_last)
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(data_loader_flag)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    data_loader_flag))
    return train_loader, val_loader, test_loader


def prepare_data(dataset, data_dir, transforms, data_loader, loader_workers,
                 train_batch_size, val_batch_size, drop_last, test_only,
                 train_sampler=None):
 
    train_transf, val_transf, test_transf = prep_data_transforms(transforms)
    train_set, val_set, test_set = prep_dataset(
        dataset, data_dir, train_transf, val_transf, test_transf, test_only)
    train_loader, val_loader, test_loader = prep_data_loaders(
        train_set, val_set, test_set, dataset, data_loader, train_batch_size,
        val_batch_size, loader_workers, drop_last, test_only, train_sampler)
    
    return train_loader, val_loader, test_loader, train_set


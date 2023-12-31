from .utils import data_transforms
import torchvision
import os

def dataset(dataset, data_dir="/cyh/cyhDataSets"):

    train_transform, valid_transform = data_transforms(dataset)
    if dataset == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(data_dir, 'fmnist'), train=True,
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.FashionMNIST(root=os.path.join(data_dir, 'fmnist'), train=False,
                                              download=True, transform=valid_transform)
        n_class = 10
        in_features=1
    
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=os.path.join(data_dir, 'mnist'), train=True,
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.MNIST(root=os.path.join(data_dir, 'mnist'), train=False,
                                              download=True, transform=valid_transform)
        n_class = 10
        in_features=1
    elif dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=os.path.join(data_dir, 'svhn'), split='train',
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.SVHN(root=os.path.join(data_dir, 'svhn'), split='test',
                                              download=True, transform=valid_transform)
        n_class = 10
        in_features=3
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        n_class = 10
        in_features=3
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        n_class = 100
        in_features=3
    elif dataset == 'imagenet':
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'imagenet-1k/train'),
                                                    transform=train_transform)
        valset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'imagenet-1k/val'),
                                                    transform=valid_transform)
        n_class = 1000
        in_features=3
    
    return trainset, valset, in_features, n_class
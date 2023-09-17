import time
import torchvision.transforms as transforms
import numpy as np
import torch
import os

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def data_transforms(dataset):
    if dataset == 'fmnist':
        MEAN = [0.5]
        STD = [0.5]
    elif dataset == 'mnist':
        MEAN = [0.1307]
        STD = [0.3081]
    elif dataset == 'svhn':
        MEAN = [0.5,0.5,0.5]
        STD = [0.5,0.5,0.5]
    elif dataset == 'cifar10':
        MEAN = [0.4913, 0.4821, 0.4465]
        STD = [0.2023, 0.1994, 0.2010]
    elif dataset == 'cifar100':
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2673, 0.2564, 0.2762]
    elif dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if  (dataset== 'fmnist'):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    
    elif  (dataset== 'mnist'):
        train_transform = transforms.Compose([
            # If the size is too small, for vgg, after poolings, it will be error
            transforms.Resize(32),
            transforms.ToTensor(), 
            transforms.Normalize(MEAN, STD)
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    elif  (dataset== 'svhn'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


    elif (dataset== 'imagenet'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif (dataset == 'cifar10'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    return train_transform, valid_transform
import PIL
import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10Sup(CIFAR10):
    """
    CIFAR10 subclass to extract a subset of samples (sup_num) for supervised training.

    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sup_num=4000, val_num=1000, random_seed=89):
        super(CIFAR10Sup, self).__init__(root, train=train, transform=transform,
                                         target_transform=target_transform, download=download)

        idx = np.random.RandomState(
            seed=random_seed).permutation(self.__len__())

        self.data = self.data[idx[:sup_num]]
        self.targets = np.array(self.targets)[idx[:sup_num]]
        self.transform = transform[0]

        
class CIFAR10Unsup(CIFAR10):
    """
    CIFAR10 subclass to extract a subset of samples (sup_num) for unsupervised training.
        - Each sample is subject to two differnt sets of transformations

    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sup_num=4000, val_num=1000, random_seed=89):
        super(CIFAR10Unsup, self).__init__(root, train=train, transform=transform,
                                         target_transform=target_transform, download=download)

        idx = np.random.RandomState(
            seed=random_seed).permutation(self.__len__())

        self.data = self.data[idx[sup_num:-val_num]]
        self.targets = None

        self.transform_unsup = transform[0]
        self.transform_unsup_aug = transform[1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img_unsup, img_unsup_aug) obtained from two different sets of transformations on self.data[idx]
        """
        img = self.data[index]

        img = PIL.Image.fromarray(img)

        if self.transform_unsup is not None:
            img_unsup = self.transform_unsup(img)
        
        if self.transform_unsup_aug is not None:
            img_unsup_aug = self.transform_unsup_aug(img)

        return img_unsup, img_unsup_aug


class CIFAR10Val(CIFAR10):
    """
    CIFAR10 subclass to extract a subset of samples (sup_num) for validation.
        
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sup_num=4000, val_num=1000, random_seed=89):
        super(CIFAR10Val, self).__init__(root, train=train, transform=transform,
                                         target_transform=target_transform, download=download)

        idx = np.random.RandomState(
            seed=random_seed).permutation(self.__len__())

        self.data = self.data[idx[-val_num:]]
        self.targets = np.array(self.targets)[idx[-val_num:]]
        self.transform = transform[0]

import os
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import tqdm
from enum import Enum
import PIL.Image

__all__ = ['DatasetType', 'DATASETS', 'CIFAR10Dataset', 'NIHDataset', 'SVHNDataset','Valid_NIHDataset']


class NIHDataset(Dataset):
    def __init__(self, image_root, split_root, split, transform=None, cache_data=False):
        super().__init__()

        self._image_root = image_root
        self._transform = transform
        self.split = split

        split_info_path = os.path.join(split_root, self.split)
        with open(split_info_path) as f_in:
            self._image_filenames = [filename.strip() for filename in f_in.readlines()]

        self._cached_images = {}
        if cache_data:
            self._cache_data = False
            print('Loading dataset ... ')
            for index in tqdm.tqdm(range(len(self))):
                self._cached_images[index] = self[index]
        self._cache_data = cache_data

    def __getitem__(self, index):
        if self._cache_data:
            return self._cached_images[index]
        else:
            image_path = os.path.join(self._image_root,self._image_filenames[index].split(' ')[0])
            label = float(self._image_filenames[index].split(' ')[1])
            image = PIL.Image.open(image_path)
            if self._transform is not None:
                image = self._transform(image)
            return [image,label]#,image_path

    def __len__(self):
        return len(self._image_filenames)

class Valid_NIHDataset(Dataset):
    def __init__(self, image_root, split_root, split, transform=None, cache_data=False):
        super().__init__()

        self._image_root = image_root
        self._transform = transform

        split_info_path = os.path.join(split_root, split)
        with open(split_info_path) as f_in:
            self._image_filenames = [filename.strip() for filename in f_in.readlines()]

        self._cached_images = {}
        if cache_data:
            self._cache_data = False
            print('Loading dataset ... ')
            for index in tqdm.tqdm(range(len(self))):
                self._cached_images[index] = self[index]
        self._cache_data = cache_data

    def __getitem__(self, index):
        if self._cache_data:
            return self._cached_images[index]
        else:
            image_path = os.path.join(self._image_root, self._image_filenames[index])
            image = PIL.Image.open(image_path)
            if self._transform is not None:
                image = self._transform(image)
            return image,image_path

    def __len__(self):
        return len(self._image_filenames)

class DatasetType(Enum):
    cifar10 = 'cifar10'
    camelyon16 = 'camelyon16'
    nih = 'nih'
    svhn = 'svhn'
    val_nih = 'val_nih'

DATASETS = {
    DatasetType.cifar10: CIFAR10Dataset,
    DatasetType.camelyon16: Camelyon16Dataset,
    DatasetType.nih: NIHDataset,
    DatasetType.svhn: SVHNDataset,
    DatasetType.val_nih: Valid_NIHDataset
}

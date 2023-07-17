from PIL import ImageOps
from torchvision import transforms
import numpy as np
from anomaly_detection.utils.datasets import DatasetType
from PIL import Image
import cv2

class EqualizeHistogram(object):
    def __call__(self, img):
        return ImageOps.equalize(img, mask=None)

class Gauss_noise(object):
    def __call__(self,img):
        img = np.asarray(img,np.float)
        img = cv2.GaussianBlur(img,(5,5),15)
        img = Image.fromarray(np.uint8(img))
        return img


class CIFAR10Transform(object):
    def __init__(self, to_grayscale=False, equalize_hist=False, to_tensor=True, normalize=True):
        tr = []
        if to_grayscale:
            tr += [transforms.Grayscale()]

        if equalize_hist:
            tr += [EqualizeHistogram()]

        if to_tensor:
            tr += [transforms.ToTensor()]

        if normalize:
            if to_grayscale:
                tr += [transforms.Normalize((0.5,), (0.5,))]
            else:
                tr += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(tr)

    def __call__(self, img):
        return self.transform(img)


class Camelyon16Transform(object):
    def __init__(self, crop_size=128, random_crop=False, random_flip=False, equalize_hist=False,
                 to_tensor=True, normalize=True):
        tr = []

        if random_crop:
            tr += [transforms.RandomCrop(crop_size)]
        else:
            tr += [transforms.CenterCrop(crop_size)]

        if equalize_hist:
            tr += [EqualizeHistogram()]

        if random_flip:
            tr += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        if to_tensor:
            tr += [transforms.ToTensor()]

        if normalize:
            tr += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(tr)

    def __call__(self, img):
        return self.transform(img)


class NIHTransform(object):
    def __init__(self, crop_size=768, resize=128, equalize_hist=False, to_tensor=True, normalize=True):
        tr = [
            transforms.Grayscale(),
            transforms.CenterCrop((crop_size, crop_size)),
        ]

        if equalize_hist:
            tr += [EqualizeHistogram()]

        tr += [transforms.Resize((resize, resize))]
        
        # if np.random.rand() > 0.2:
        #     tr += [Gauss_noise()]

        if to_tensor:
            tr += [transforms.ToTensor()]

        if normalize:
            tr += [transforms.Normalize((0.5,), (0.5,))]

        self.transform = transforms.Compose(tr)

    def __call__(self, img):
        return self.transform(img)


TRANSFORMS = {
    DatasetType.cifar10: CIFAR10Transform,
    DatasetType.camelyon16: Camelyon16Transform,
    DatasetType.nih: NIHTransform,
    DatasetType.svhn: CIFAR10Transform,
    DatasetType.val_nih: NIHTransform,
}

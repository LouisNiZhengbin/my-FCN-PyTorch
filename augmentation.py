import torch
import numpy as np
from torchvision import transforms, utils
from torchvision.transforms import functional as F
import random


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

        self.padding = padding
        self. pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(image, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = image.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


    def __call__(self, sample):
        items = [sample['image']]
        if 'label' in sample.keys():
            items.append(sample['label'])

        for item in items:
            if self.padding is not None:
                item = F.pad(item, self.padding)

            ## padding the height if needed
            if self.pad_if_needed and image[1] < self.size[0]:
                item = F.pad(item, (0, self.size[0] - image[1]))
            ## padding the width if needed
            if self.pad_if_needed and image[0] < self.size[1]:
                item = F.pad(item, (self.size[0] - image[1], 0))

        i, j, th, tw = self.get_params(items[0], self.size)

        if 'label' in sample.keys():
            return{'image': F.crop(items[0], i, j, th, tw), 'label': F.crop(items[1], i, j, th, tw)}
        else: 
            return{'image': F.crop(items[0], i, j, th, tw)}









class ToTensor(object):
    """Convert 2 (sample['image'], sample['label']) in  ``PIL Image``s or ``numpy.ndarray``s to tensors.

    Converts  PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """


    def __call__(self, sample):
        image = F.to_tensor(sample['image'])
        if 'label' in sample.keys():
            label = F.to_tensor(sample['label'])
            label = (255. * label).long()
            # label = np.array(sample['label'])
            # label = label.reshape((1, label.shape[0], label.shape[1]))
            return {'image': image, 'label': label.squeeze()}
        else:
            return {'image': image}



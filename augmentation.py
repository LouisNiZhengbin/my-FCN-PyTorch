from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top+new_h, left: left+new_w]
        label = label[top: top+new_h, left: left+new_w]

        return {'image':image, 'label':label}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}



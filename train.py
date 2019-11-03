import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import time
import argparse
import shutil
import os

from fcn32s import FCN32s
from metrics import BCEDiceLoss, MetricTracker, dice_coeff
from utils import save_checkpoints
from options import Options
from dataset import PascalVOC
from augmentation import ToTensor, RandomCrop


class Trainer(object):
	def __init__(self, args):
		self.args = args
		self.device = args.device
		self.epochs = args.epochs

		# Networks
		self.model = FCN32s()

		# Criterion
		self.criterion = 

		# Optimizer
		self.lr = args.lr
		self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
		self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=18, gamma=0.1)

		# Dataset
		self.train_set = PascalVOC(transforms=transforms.Compose([
										RandomCrop(200),
										ToTensor()
									]))

		self.valid_set = PascalVOC(names_file='./VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
									transforms=transforms.Compose([
											RandomCrop(200),
											ToTensor()
									]))

		self.train_loader = DataLoader(self.train_set,
										batch_size=args.batch_size,
										num_workers=2,
										shuffle=True)
		self.valid_loader = DataLoader(self.valid_set,
										batch_size=1,
										num_workers=2,
										shuffle=False)

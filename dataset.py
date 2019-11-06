import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



import os
from augmentation import RandomCrop, ToTensor
from fcn_utils import show_batch, show_dataset



from PIL import Image

class PascalVOC(Dataset):

	def __init__(self,
				root_dir='./VOCdevkit/VOC2012',
				img_dir ='./VOCdevkit/VOC2012/JPEGImages/',
				img_label_dir='./VOCdevkit/VOC2012/SegmentationClass/',
				names_file='./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
				transforms=None):


		self.root_dir = root_dir
		self.img_dir = img_dir
		self.img_label_dir = img_label_dir
		self.names_file = names_file
		self.transforms = transforms
		self.size = 0
		self.names_list = []

		if not os.path.isfile(self.names_file):
			print(self.names_file + ' does not exist!')
		with open(self.names_file) as file:
			for f in file:
				self.names_list.append(f[:-1])
				self.size += 1

	def __getitem__(self, idx):
		img_path = self.img_dir + self.names_list[idx] + '.jpg'
		img_label_path = self.img_label_dir + self.names_list[idx] + '.png'

		if not os.path.isfile(img_path):
			print(img_path + ' does not exist!')
			return None
		if not os.path.isfile(img_label_path):
			print(img_label_path + ' does not exist!')
			return None

		image = Image.open(img_path)
		label = Image.open(img_label_path)
		sample = {'image': image, 'label': label}
		
		if self.transforms:
			sample = self.transforms(sample)

		return sample

	def __len__(self):
		return self.size


if __name__ == '__main__':
	train_dataset = PascalVOC(transforms=transforms.Compose([RandomCrop(250), ToTensor()]))
	# from fcn_utils import show_map
	# print(len(train_dataset))
	# for i in range(1225, 1230):
	# 	sample = train_dataset[i]
	# 	show_map(sample)
	# 	plt.show()

	train_dataloader = DataLoader(train_dataset, batch_size=4)
	show_batch(train_dataloader, 4)





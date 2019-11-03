### parameters initilization
import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def weights_init(Model):
	for m in Model.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight)
			if m.bias is not None:
				m.bias.data.zero_()
		if isinstance(m, nn.ConvTranspose2d):
			nn.init.kaiming_normal_(m.weight)

def _show_batch(sample_batch):

	image_batch, label_batch = sample_batch['image'], sample_batch['label']
	batch_size = len(image_batch)

	grid1 = utils.make_grid(image_batch)
	grid2 = utils.make_grid(label_batch)

	plt.subplot(2,1,1)
	plt.imshow(grid1.numpy().transpose((1, 2, 0)))
	plt.subplot(2,1,2)
	plt.imshow(grid2.numpy().transpose((1, 2, 0)))
	plt.show()

def show_batch(dataloader, num_batch):
	plt.figure()
	for cnt, batch in enumerate(dataloader):
		_show_batch(batch)

		if cnt == num_batch:
			break


def show_dataset(dataset, nums):
	plt.figure()
	for cnt, data in enumerate(dataset):

		image = data['image']
		label = data['label']

		plt.subplot(2,2,1)
		plt.imshow(label)
		plt.subplot(2,2,2)
		plt.imshow(image)
		plt.show()

		if cnt == nums:
			break


                
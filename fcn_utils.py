### parameters initilization
import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def weights_init(Model):
	for m in Model.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight)
			if m.bias is not None:
				m.bias.data.zero_()
		if isinstance(m, nn.ConvTranspose2d):
			nn.init.kaiming_normal_(m.weight)

def show_batch(dataloader, num_batch):
	plt.figure()
	for cnt, batch in enumerate(dataloader):
		image_batch, label_batch = batch['image'], batch['label']
		batch_size = len(image_batch)

		grid1 = utils.make_grid(image_batch)
		grid2 = utils.make_grid(label_batch)

		plt.subplot(2,1,1)
		plt.imshow(grid1.numpy().transpose((1, 2, 0)))
		plt.title(f'image_batch {cnt+1}')
		plt.subplot(2,1,2)
		plt.imshow(grid2.numpy().transpose((1, 2, 0)))
		plt.title(f'label_batch {cnt+1}')
		plt.show()
		# np.set_printoptions(threshold=np.inf)
		# print(grid2)

		if cnt == num_batch:
			break



def show_dataset(sample, axis=None):
    """
    Return an image with the shape mask if supplied
    """
    if axis:
        axis.imshow(sample['image'])
        plt.show()
        if sample['label'] is not None:
            axis.imshow(sample['label'])
            plt.show()
    else:
        plt.imshow(sample['image'])
        plt.show()
        if sample['label'] is not None:
            plt.imshow(sample['label'])
            plt.show()


def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=''):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model) 


                
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
from fcn_utils import save_checkpoints
#from options import Options
from dataset import PascalVOC
from augmentation import ToTensor, RandomCrop


class Trainer(object):
	def __init__(self):
		self.device = 'cpu'
		self.epochs = 2

		# Networks
		self.model = FCN32s(21)

		# Criterion
		self.criterion = nn.CrossEntropyLoss(ignore_index=0)

		# Optimizer
		self.lr = 1000
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
										batch_size=4,
										num_workers=2,
										shuffle=True)
		self.valid_loader = DataLoader(self.valid_set,
										batch_size=4,
										num_workers=2,
										shuffle=False)


	def train(self):
		best_loss = 1.0
    
		tt = time.time()
		self.model.train()        
		self.model.to(self.device)            

		print("Start training ...")

		for epoch in range(self.epochs):
			print(f'Epoch {epoch+1}/{self.epochs}')

			train_metrics = self._train(epoch)
			valid_metrics = self._valid(epoch)

            # store best loss and save a model checkpoint
        
			is_best = valid_metrics['loss'] < best_loss
        
			state = {'epoch': epoch,
                     'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'best_loss': best_loss}
                    
			save_checkpoints(state, is_best)

		time_ellapsed = time.time() - tt
		print('Training complete in {time_ellapsed // 60}m {time_ellapsed % 60}s')


	def _train(self, epoch):

		for idx, batch in enumerate(self.train_loader):

			image = batch['image'].to(self.device)
			label = batch['label'].to(self.device)

			self.optimizer.zero_grad()

			out = self.model(image)


			loss = self.criterion(out, label)
			print(f'loss: {loss}')

			loss.backward()

			self.optimizer.step()
			self.lr_scheduler.step()

	def _valid(self, epoch):
		valid_loss = MetricTracker()
    
		valid_acc  = MetricTracker()

		self.model.eval()

        # Iterate over data
		for idx, sample in enumerate(self.valid_loader):
            # get the inputs
			image = sample['image'].to(self.device)
			label = sample['label'].to(self.device)

            # forward
			out  = self.model(image)
			loss = self.criterion(out, label)
        

			valid_acc.update(dice_coeff(out, label), out.shape[0])
			valid_loss.update(loss.item(), out.shape[0])
        
		print(f'Valid Loss: {valid_loss.avg:.4f} Acc: {valid_acc.avg:.4f}')
    
		self.model.train()        
		return {'loss': valid_loss.avg, 'acc': valid_acc.avg}

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

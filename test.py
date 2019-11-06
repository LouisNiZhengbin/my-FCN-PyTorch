import torch
import torch.nn as nn
from dataset import PascalVOC
from augmentation import RandomCrop, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from fcn32s import FCN32s

fcn = FCN32s(21)
train_dataset = PascalVOC(transforms=transforms.Compose([RandomCrop(200), ToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=2)

criterion = nn.CrossEntropyLoss(ignore_index=255)
for batch in train_dataloader:
	print(batch['image'].size())
	print(batch['label'].size())
	print(batch['image'].dtype)
	print(batch['label'].dtype)	
	out = fcn(batch['image'])
	loss = criterion(out, batch['label'])
	print(loss)
	break
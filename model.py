## The Model of FCN
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class FCN32s(nn.Module):
	def __init__(self, n_class):
		super(FCN32s, self).__init__()

		# original input: (n, 3, w, h)

		# conv1 layer
		self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100) ##(n, 64, w + 198,h + 198)
		self.relu1_1 = nn.ReLU()
		self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
		self.relu1_2 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2, 2)  #(n, 64, w/2, h/2)
		##(n, 64, (w+198)/2, (h+198)/2)

		# conv2 layer
		self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
		self.relu2_1 = nn.ReLU()
		self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
		self.relu2_2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2, 2) #(n, 128, w/4, h/4)
		##(n, 128, (w+198)/4, (h+198)/4)

		# conv3 layer
		self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
		self.relu3_1 = nn.ReLU()
		self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_2 = nn.ReLU()
		self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(2, 2) #(n, 258, w/8, h/8)
		##(n, 258, (w+198)/8, (h+198)/8)

		# conv4 layer
		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.relu4_1 = nn.ReLU()
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_2 = nn.ReLU()
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_3 = nn.ReLU()
		self.pool4 = nn.MaxPool2d(2, 2) #(n, 512, w/16, h/16)
		##(n, 512, (w+198)/16, (h+198)/16)

		# conv5 layer
		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_1 = nn.ReLU()
		self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_2 = nn.ReLU()
		self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_3 = nn.ReLU()
		self.pool5 = nn.MaxPool2d(2, 2) #(n, 512, w/32, h/32)
		##(n, 512, (w+198)/32, (h+198)/32)

		# fc6
		self.fc6 = nn.Conv2d(512, 4096, 7)
		self.relu6 = nn.ReLU(inplace=True)
		self.drop6 = nn.Dropout2d() #(n, 4096, w/32-6, h/32-6)
		##(n, 4096, (w+6)/32, (h+6)/32)


		self.fc7 = nn.Conv2d(4096, 4096, 1)
		self.relu7 = nn.ReLU(inplace=True)
		self.drop7 = nn.Dropout2d() #(n, 4096, w/32-6, h/32-6)
		##(n, 4096, (w+6)/32, (h+6)/32])

		self.score_fr = nn.Conv2d(4096, n_class, 1) #(n, n_class, w/32-6, h/32-6, n_class) ##(n, (w+6)/32, (h+6)/32)
													##(n, n_class, (w+6)/32, (h+6)/32)
		self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False) #(n, n_class, (w/32-6 + 1)*32, (h/32-6 + 1)*32, n_class) = (n, w - 160, h - 160)
													##(n, n_class, w + 48, h+48)

	def forward(self, x):
		

		h = self.relu1_1(self.conv1_1(h))
		h = self.relu1_2(self.conv1_2(h))
		h = self.pool1(h) 

		h = self.relu2_1(self.conv2_1(h))
		h = self.relu2_2(self.conv2_2(h))
		h = self.pool2(h) 

		h = self.relu3_1(self.conv3_1(h))
		h = self.relu3_2(self.conv3_2(h))
		h = self.relu3_3(self.conv3_3(h))
		h = self.pool3(h) 
		pool3_output = h #used in fcn8s

		h = self.relu4_1(self.conv4_1(h))
		h = self.relu4_2(self.conv4_2(h))
		h = self.relu4_3(self.conv4_3(h))
		h = self.pool4(h) 
		pool4_output = h #used in fcn16s

		h = self.relu5_1(self.conv5_1(h))
		h = self.relu5_2(self.conv5_2(h))
		h = self.relu5_3(self.conv5_3(h))
		h = self.pool5(h) 
		pool5_output = h #used in fcn16s

		h = self.relu6(self.fc7(h))
		h = self.drop6(h) 

		h = self.score_fr(h) 

		h = self.upscore(h) 

		h = h[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()
		return h



if __name__ == '__main__':
	fcn = FCN32s(20)
	print(fcn)





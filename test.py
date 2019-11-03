# test for fcn
from dataset import PascalVOC
train_dataset = PascalVOC(root_dir='./VOCdevkit/VOC2012',
							  img_dir ='./VOCdevkit/VOC2012/JPEGImages',
							  img_label_dir='./VOCdevkit/VOC2012/SegmentationClass',
							  names_file='./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')

plt.figure()
for (cnt,i) in enumerate(train_dataset):
	image = i['image']
	label = i['label']
		
	ax = plt.subplot(4, 4, cnt+1)
	ax.axis('off')
	ax.imshow(image)
	ax.imshow(label)
	plt.pause(0.001)

	if cnt == 15:
		break
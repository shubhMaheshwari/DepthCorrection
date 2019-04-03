# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# Get the Hyperparaeters 
opt = TestOptions().parse()

target_dataset = DataSet(opt,'/media/shubh/PranayHDD/Kinect/')
target_loader = torch.utils.data.DataLoader(target_dataset,batch_size=opt.batch_size,num_workers=opt.num_workers,shuffle=False)

device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device
model = Model(opt)
if opt.use_gpu:
	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Load the weights and make predictions
model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.load_epoch)))

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')

model.eval()

if not os.path.exists(opt.test_results):
	os.mkdir(opt.test_results)

def save_results(index,rgb_img,org_depth_img,corrupted_depth_img,pred_depth_img):

	fig = plt.figure(figsize=(16,8))
	# RGB
	ax = fig.add_subplot(1,4,1)
	ax.imshow(rgb_img.T[::-1,::-1,:])
	ax.axis('off')
	ax.set_title('RGB Images')
	# Corrupted Depth
	ax = fig.add_subplot(1,4,2)
	ax.imshow(corrupted_depth_img[::-1,::-1].T,cmap='gray')
	ax.axis('off')
	ax.set_title('Corrupted Depth Images')

	# Orignal Depth
	ax = fig.add_subplot(1,4,3)
	ax.imshow(org_depth_img[::-1,::-1].T,cmap='gray')
	ax.axis('off')
	ax.set_title('Original Depth Images')
	
	# Predicted Depth
	ax = fig.add_subplot(1,4,4)
	ax.imshow(pred_depth_img[::-1,::-1].T,cmap='gray')
	ax.axis('off')
	ax.set_title('Predicted Depth Images')

	plt.savefig(os.path.join(opt.test_results,'{}.png'.format(index)))
	print("Saved Image")


for j,(rgb_images,depth_images, original_depth,corrupted_depth) in enumerate(target_loader):		
	try:
		pred_depth = model(rgb_images.to(device),depth_images.to(device))
		for i in range(rgb_images.shape[0]):
			save_results(j*rgb_images.shape[0]+i, rgb_images[i,:,-1,:,:].cpu().data.numpy(),original_depth[i,:,:].cpu().data.numpy(),corrupted_depth[i,:,:].cpu().data.numpy() , pred_depth[i,:,:].cpu().data.numpy())
	except Exception  as r:
		print("Error:",r)
		torch.cuda.empty_cache()
		continue





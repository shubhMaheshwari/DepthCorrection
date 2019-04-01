# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
from sklearn.metrics import confusion_matrix,roc_curve
import cv2
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import pandas as pd
# Get the Hyperparaeters 
opt = TestOptions().parse()

target_dataset = DataSet(opt,'./test_WyRytb0.csv')
target_loader = torch.utils.data.DataLoader(target_dataset,batch_size=60,num_workers=30,shuffle=False)

# Load the model and send it to gpu
test_transforms =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
						 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
	transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
						 std = [ 1., 1., 1. ]) ])


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
K = opt.best_k

def get_accuracy(pred_sample):
	pred_ind = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )
	confidence = pred_sample[np.arange(len(pred_ind)),pred_ind]

	return pred_ind,confidence

for target_images,file_names in target_loader:		
	try:
		pred_target = model(target_images.to(device))
	except RuntimeError  as r:
		print("Error:",r)
		torch.cuda.empty_cache()
		continue

	target_pred,confidence = get_accuracy(pred_target)
	for i,file_name in enumerate(file_names):
		ind = target_dataset.filename2id[file_name]
		if target_dataset.dataframe['image_name'][ind] != file_name:
			print("Wrong filename:",target_dataset.dataframe['image_name'][ind],file_name)

		target_dataset.dataframe['label'][ind] = int(target_pred[i])  
		print(file_name,confidence)
target_dataset.dataframe.to_csv('./test.csv',index=False,sep=',')




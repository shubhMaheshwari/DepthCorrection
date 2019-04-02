# Main module to train the model, load the data,
# do gradient descent etc. followed by saving our model for later testing
from dataloader import DataSet,create_samplers
from model import Model
from visualizer import Visualizer 
from options import TrainOptions
import torch
from torchvision.transforms import *
import torch.optim as optim
import numpy as np 
import os
# Get the Hyperparaeters 
opt = TrainOptions().parse()

sample_dataset = DataSet(opt,"/media/shubh/PranayHDD/Kinect/")
train_sampler,val_sampler = create_samplers(sample_dataset.__len__(),opt.split_ratio)
data_loader = torch.utils.data.DataLoader(sample_dataset,sampler=train_sampler,batch_size=opt.batch_size,num_workers=opt.num_workers)
data_val_loader = torch.utils.data.DataLoader(sample_dataset,sampler=val_sampler,batch_size=opt.val_batch_size,num_workers=0,shuffle=False)


# Check if gpu available or not
device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device

# Load the model and send it to gpu
model = Model(opt)
model = model.to(device)
if opt.use_gpu:	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')	


# Make checkpoint dir to save best models
if not os.path.exists('./checkpoints'):
	os.mkdir('./checkpoints')

# If require load old weights
if opt.load_epoch > 0:
	model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.load_epoch)))

else:
	def init_weights(m):
		if type(m) == torch.nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)
		elif isinstance(m,torch.nn.Conv3d):
			torch.nn.init.xavier_uniform_(m.weight,gain=np.sqrt(2))
	model.apply(init_weights)

if opt.display:
	vis = Visualizer(opt)
else:
	vis = None
	
# Loss functons(Cross Entropy)
# Adam optimizer
criterion_sample = torch.nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)

def save_model(model,epoch):
	filename = './checkpoints/' + 'model_{}.pt'.format(epoch)
	torch.save(model.state_dict(), filename)

loss_list = []

# Training loop
for epoch in range(opt.epoch):
	# In each epoch first trained on images and then perform validation

	model.train()
	for i, (rgb_images,depth_images, original_depth,corrupted_depth) in enumerate(data_loader):			
		# Do a prediction

		try:
			optimizer.zero_grad()
			pred_depth = model(rgb_images.to(device),depth_images.to(device))
			# Calculate loss
			loss = criterion_sample( pred_depth, (original_depth).to(device))
		except Exception as e:
			print("Error:",e)	
			torch.cuda.empty_cache()
			continue

		# Do backpropogation followed by a gradient descent step
		loss.backward()
		optimizer.step()	

		# # Once in a while print losses and accuracy
		if i % opt.print_iter == 0:
			# print(pred_sample[4,:].cpu().data.numpy(),labels[4].numpy())
			# Print loss
			# print(pred_depth)
			# print(original_depth)
			print("Iter:{}/{} Loss:{}".format(i,80, loss.cpu().data.numpy()))
			loss_list.append(loss.cpu().data.numpy())
			if opt.display:
				vis.plot_loss(loss_list)
				vis.show_image(rgb_images[0,:,-1,:,:].cpu().data.numpy(),original_depth[0,:,:].cpu().data.numpy(),corrupted_depth[0,:,:].cpu().data.numpy() , pred_depth[0,:,:].cpu().data.numpy(),2,"Predicted Depth")

	# Validate model using the validation set
	model.eval()

	rgb_images,depth_images, original_depth,corrupted_depth = next(iter(data_val_loader))

	try:
		pred_depth = model(rgb_images.to(device),depth_images.to(device))
		# Calculate loss
		loss = criterion_sample(pred_depth,original_depth.to(device))	
	except RuntimeError  as r:
		torch.cuda.empty_cache()
		continue

	print("Validation:{}th epoch Loss:{}".format(epoch,loss))

	if epoch%10 ==0:
		save_model(model,epoch)

	# Update lr 
	if epoch > opt.lr_decay_iter:
		for g in optimizer.param_groups:
			g['lr'] = opt.lr_decay_param*g['lr']

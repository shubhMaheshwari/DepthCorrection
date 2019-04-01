import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import scipy.ndimage

class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset
	"""
	def __init__(self,opt, dirpath):
		"""
		Initializes the dataset for loading depth and RGB images

		param @opt: options for running the model
		param @dirpath: options for running the model
		"""
		super().__init__()

		self.dir = dirpath

		# Store the filenames of all rgb and depth images avaiable
		sequences = [dirname for dirname in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,dirname))] # Find all directories 

		total_rgb_images = [] # Store all rgb images here
		total_depth_images = [] # Store all depth images here

		# Loop over all directories and save their paths
		for dirname in sequences:
			depth_images = []
			rgb_images = []
			for i in range(1,opt.total_sequence_length+1):
				depth_image_path = os.path.join(dirpath,dirname,'depth_003324244847_{}.png'.format(i))
				rgb_image_path = os.path.join(dirpath,dirname,'rgb_003324244847_{}.png'.format(i))
				if os.path.isfile(depth_image_path) and os.path.isfile(depth_image_path):
					depth_images.append(depth_image_path)
					rgb_images.append(rgb_image_path)

			total_depth_images.append(depth_images)
			total_rgb_images.append(rgb_images)
		# Copy to class variables
		self.total_depth_images = total_depth_images
		self.total_rgb_images = total_rgb_images
		
		# Store the number of directories
		self.total_sequences = len(sequences)
		self.images_per_sequence = opt.total_sequence_length - opt.model_sequence_length + 1

		# Store options for future use
		self.opt = opt

		# RGB transformations 
		self.rgb_tranform = transforms.Compose([
			transforms.Resize((424, 512)),
			transforms.ToTensor()
			])
		print("Dataset Total sequences:",self.total_sequences)
		print("Images per sequences:",self.images_per_sequence)
	def corrupt_depth(self,depth_image):
		"""
			Takes a depth image and corrupts it using dialation:
		"""
		corrupted_depth = scipy.ndimage.grey_dilation(depth_image, size=(5,5), structure=np.ones((5,5)))

		return (corrupted_depth/255.0).astype('float32')

	def __getitem__(self,idx):
	
		"""
			Given an index returns the data 

			returns: 
				rgb_images: torch.Tensor containing rgb images from t to t+k 
				depth_images: torch.Tensor contains corrupted depth images from t to t+k 
				original depth: torch.Tensor contaning the depth image(t+k) that needs to be predicted by the model
		"""
		idx = idx % 2

		sequence_id = idx // self.images_per_sequence
		frame_id = idx % self.images_per_sequence

		# Load RGB and corrupted Depth
		rgb_list = []
		depth_list = []
		for i in range(self.opt.model_sequence_length):
			rgb_path = self.total_rgb_images[sequence_id][frame_id + i] 
			rgb_im = Image.open(rgb_path)
			rgb_list.append(self.rgb_tranform(rgb_im))
			depth_path = self.total_depth_images[sequence_id][frame_id + i]
			depth_im = np.array(Image.open(depth_path))
			corrupted_depth_im = self.corrupt_depth(depth_im)
			depth_list.append(corrupted_depth_im)

		rgb_list = torch.stack(rgb_list).transpose(0,1)
		depth_list = torch.tensor(np.array(depth_list)).unsqueeze(0)

		return rgb_list,depth_list, torch.tensor((depth_im/255.0).astype('float32')), torch.tensor(corrupted_depth_im)

	def __len__(self):
		return self.total_sequences*self.images_per_sequence # number of sequences * number of frame in each sequence


def create_samplers(length,split):
	"""
		To make a train and validation split 
		we must know out of which indices should
		the dataloader load for training images and validation images
	"""

	# Validation dataset size
	val_max_size = np.floor(length*split).astype('int')

	# List of Randomly sorted indices
	idx = np.arange(length)
	idx = np.random.permutation(idx)

	# Make a split
	train_idx = idx[0:val_max_size]
	validation_idx = idx[val_max_size:length]

	# Create the sampler required by dataloaders
	train_sampler = SubsetRandomSampler(train_idx)
	val_sampler = SubsetRandomSampler(validation_idx)

	return train_sampler,val_sampler


if __name__ == "__main__":
	pass
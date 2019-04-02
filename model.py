import torch 
import torch.nn as nn
import torchvision.models as models

class C3D(nn.Module):
	"""
	The C3D network as described in [1].
	"""

	def __init__(self,opt,input_channel):
		super(C3D, self).__init__()

		self.get_features = nn.Sequential(
		nn.Conv3d(input_channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(64),

		nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

		nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(128),

		nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

		nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(256),

		nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(256),
		
		nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

		nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(512),
		
		nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(512),
		
		nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

		nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(512),

		nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
		nn.LeakyReLU(0.2,inplace=True),
		nn.BatchNorm3d(512),
		
		

		nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
		)

	def forward(self, x):
		h = self.get_features(x)
		return h


class Decoder(nn.Module):
	"""
		Decoder module takes in a embedding of the rgb and corrupted depth images and recreates the rgb image
	"""
	def __init__(self,opt):
		super(Decoder,self).__init__()
		self.opt = opt

		self.get_image = nn.Sequential(
		nn.ConvTranspose2d(1024, 512, 3,stride=2, padding=1),
		nn.LeakyReLU(0.2,inplace=True),

		nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1),
		nn.LeakyReLU(0.2,inplace=True),

		nn.ConvTranspose2d(512, 512, 3, padding=1),
		nn.LeakyReLU(0.2,inplace=True),

		nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,output_padding=(1,0)),
		nn.LeakyReLU(0.2,inplace=True),


		nn.ConvTranspose2d(256, 256, 3, padding=1),
		nn.LeakyReLU(0.2,inplace=True),
		
		nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,output_padding=(1,0)),
		nn.LeakyReLU(0.2,inplace=True),

		nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,output_padding=(1,0)),
		nn.LeakyReLU(0.2,inplace=True),

		nn.ConvTranspose2d(64, 1, 3, stride=1, padding=1),
		nn.LeakyReLU(0.2,inplace=True)
		)

		self.relu = nn.LeakyReLU()
	def forward(self,embedding):
		h = self.get_image(embedding)
		return h[:,:,:,:-1].squeeze(1)
		
class Model(nn.Module):
	"""
		Our model 
		It is normal CNN, we will keep on updating it until 
		we get the desired result 

		Trial 1 =>  8 layer CNN network. 
	"""
	def __init__(self,opt):
		super(Model,self).__init__()
		self.opt = opt
		self.rgb3D = C3D(opt,4)
		self.depth3D = C3D(opt,1)

		self.depth2d = Decoder(opt)

	def forward(self,RGBimages,Depthimages):
		rgb_encoding = self.rgb3D(RGBimages)   
		depth_encoding = self.depth3D(Depthimages)	
		print(rgb_encoding)
		concat_embedding = torch.cat((rgb_encoding, depth_encoding), 1)
		concat_embedding = concat_embedding.squeeze(2)

		pred_depth_image = torch.sigmoid(self.depth2d(concat_embedding))	
		return pred_depth_image
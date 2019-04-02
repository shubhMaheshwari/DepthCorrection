import argparse
import os
import torch 


class BaseOptions(object):
	"""
		Base options given to run the network
	"""
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.device = None
	def initialize(self):
		# Basic Details
		self.parser.add_argument('--batch_size',type=int, default=3, help='Batch Size for each iternations')
		self.parser.add_argument('--val_batch_size',type=int, default=1, help='Batch Size for each iternations')
		self.parser.add_argument('--gpus', default='0', help='-1: cpu else is a list of gpu ids eg. 0,1,2')
		self.parser.add_argument('--use_gpu',type=bool, default=True, help='Whether to use gpu')

		# Parameters for network
		self.parser.add_argument('--model_sequence_length',type=int, default=3, help='For the model the total number of RGDB images used')
		self.parser.add_argument('--total_sequence_length',type=int, default=200, help='total number of frames collected from kinect')


		# Loading trained network
		self.parser.add_argument('--load_epoch', type=int, default=-1, help='From checkpoints dir get the best model based on the epoch for traning and testing')
		self.parser.add_argument('--num_workers', type=int, default=2, help='From checkpoints dir get the best model based on the epoch for traning and testing')


		# Visdom settings during training
		self.parser.add_argument('--display', type=bool, default=False, help='Visualize data')
		self.parser.add_argument('--display_server', type=str, default='http://localhost', help='visdom display host')
		self.parser.add_argument('--display_port', type=int, default=8097, help='Visdom display port')


		# # Also start the visaulizer
		# os.system("python -m visdom.server &")

	def parse(self):
		"""
			Load the arguments given and processes them for further use
		"""
		self.initialize()
		opt = self.parser.parse_args()
		args = vars(opt)


		if opt.use_gpu:
			gpus = [int(i) for i in opt.gpus.split(',')]
			print("=> active GPUs: {}".format(gpus))
			opt.gpus = gpus

		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		return opt


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--mode', type=str, default="Train", help='Why are we running the model eg. Train , test, finetune etc.')
		self.parser.add_argument('--epoch', type=int, default=100, help='Number of epoch')

		# Hyperparameters
		self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
		self.parser.add_argument('--lr_decay_iter', type=int, default=30, help='learning rate')
		self.parser.add_argument('--lr_decay_param', type=float, default=0.9, help='learning rate')
		self.parser.add_argument('--weight_decay', type=float, default=0.005, help='Regularization')
		self.parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
		self.parser.add_argument('--print_iter', type=int, default=	1, help='Invervals to print between iters')

		# Train validation split
		self.parser.add_argument('--split_ratio', type=float, default=0.8, help='Number of interations per epoch')
		
class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)	
		self.parser.add_argument('--mode', type=str, default="Test", help='Why are we running the model eg. Train , test, finetune etc.')

if __name__ == "__main__":
	train_opt = TrainOptions().parse()

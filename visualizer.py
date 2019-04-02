import os
import numpy as np
import visdom
import torch
class Visualizer():
	def __init__(self, opt):

		self.opt = opt
		self.vis = visdom.Visdom()
		self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port)

	# errors: dictionary of error labels and values        
	def plot_graph(self,X,Y,labels,display_id,title,axis=['x','y']):
		Y = np.array(Y).T
		if X == None: 
			X = np.arange(0,Y.shape[0])
		
		self.vis.line(
			X=X,
			Y=Y,
			win = display_id,
			opts={
			'title': title,
			'legend': labels,
			'xlabel': axis[0],
			'ylabel': axis[1]}
			)

		return

	def plot_loss(self,loss_list):
		self.plot_graph(None,[loss_list],["Loss"] ,display_id=1,title='Loss over time',axis=['Epoch','Loss'])

	def show_image(self,rgb_img,org_depth_img,corrupted_depth_img,pred_depth_img,display_id,title="Images"):        
		print(np.mean(pred_depth_img))
		print(np.mean(org_depth_img))
		self.vis.images(
			np.vstack((org_depth_img[::-1,::-1].T,corrupted_depth_img[::-1,::-1].T )),
			win=display_id,
			nrow=2,
			opts={
			'caption': "Result",
			'title': title
			})

		self.vis.images(
			np.vstack(((corrupted_depth_img-pred_depth_img)[::-1,::-1].T, pred_depth_img[::-1,::-1].T )),
			win=display_id+1,
			nrow=2,
			opts={
			'caption': "Result",
			'title': title
			})
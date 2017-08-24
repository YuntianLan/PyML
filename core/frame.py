import numpy as np
import matplotlib.pyplot as plt

from layers.DenseLayer import *
from layers.ReLu import *
from layers.Softmax import *
from layers.SVM import *
from layers.MaxPool import *
from layers.im2col import *

import sys
sys.path.append('..')
from data.data_util import *
sys.path.pop(-1)



'''
This file contains the general framework that will integrate
all the layers.
This is a RAW draft, so there are not a lot of layers.
Just the basic and famous ones.

Layers to implement:
Learnable Layer:
	Dense Layer
	Batch Normalization Layer

	Conv Layer
	Dropout Layer (maybe)
	Pooling Layer (maybe)
Nonlinearity:
	ReLu
	Sigmoid
	Tanh
Loss Function:
	Softmax
	SVM
Gradient Decent:
	SGD
	Adam (maybe)
'''


class frame(object):

	'''
	The frame class integrates all the layers with a specific dataset
	'''

	def __init__(self,layers,data,args):
		'''
		Initialize a frame with predefined layers, data, and optional data.

		Arguments:
			layers: list of layers
			data: dictionary containing the training and validation data
				'x_train', 'y_train', 'x_val', 'y_val'
			args: dictionary containing optional arguments
				learning_rate  =   3e-3
				batch_size     =   50
				epoch          =   10
				lr_rule        =   'constant'
				update_rule    =   'sgd'
				loss_func      =   'softmax'
				reg 		   =   1e-3
				debug 		   =   0
		'''
		self.layers = layers   # Last element is loss layer
		self.x_train = data['x_train']
		self.y_train = data['y_train']
		self.x_val = data['x_val']
		self.y_val = data['y_val']

		self.learning_rate = args.get('learning_rate',3e-3)
		self.batch_size = args.get('batch_size',50)
		self.epoch = args.get('epoch',10)
		self.reg = args.get('reg',1e-3)
		self.debug = args.get('debug',0)

		# For now only does constant learning rate,
		# sgd gradient update, softmax loss function,
		# maybe add more in the future?
		assert args.get('lr_rule','constant')=='constant',\
		'only accept constant for lr_rule'
		self.lr_rule = args.get('lr_rule','constant')

		assert args.get('update_rule','sgd')=='sgd',\
		'only accept sgd for update_rule'
		self.update_rule = args.get('update_rule','sgd')

		# assert args.get('loss_func','softmax')=='softmax',\
		# 'only accept softmax for loss_func'
		# self.loss_func = args.get('loss_func','softmax')

		size = (self.batch_size,) + self.x_train[0].shape
		all_size = []
		for l in layers:
			all_size.append(size)
			size = l.init_size(size)
		all_size.append(size)
		
		print '\nInitialization Complete'

	def check_acc(self, y1, y2):
		assert y1.shape==y2.shape, 'Incompatible input'
		return len(filter(lambda x: abs(x)<1e-4, y1 - y2))/float(len(y1))

	def test_accu_loss(self, num):
		'''
		Test the frame using x_val and y_val, check the accuracy
		of prediction and loss, randomly subsample
		Returns two scalars suggesting the accuracy and loss

		TODO: maybe support multiple ways of subsampling?
		'''
		param = {'mode':'test','debug':self.debug,'reg':self.reg}


		N = self.y_val.shape[0]
		run = num / self.batch_size
		idx = np.random.choice(N, num)
		x_test = self.x_val[idx]
		y_test = self.y_val[idx]

		

		running_acc, running_loss = [], []

		for i in xrange(run):
			start = i * self.batch_size

			x_temp = x_test[start: start + self.batch_size]
			y_temp = y_test[start: start + self.batch_size]

			for l in self.layers:
				x_temp = l.forward(x_temp,param)

			running_acc.append(self.check_acc(x_temp,y_temp))
			curr_loss = self.layers[-1].getLoss(y_temp)
			
			for l in self.layers:
					w = l.get_kernel()
					if not w is None:
						curr_loss += 0.5 * self.reg * np.sum(w[:-1] * w[:-1])
						
			running_loss.append(curr_loss)
		
		return sum(running_acc) / run, sum(running_loss) / run




	def train(self, verbose=0, gap=100, val_num=None):
		'''
		Train the frame

		verbose: 0-3, level of printing the training process will do
		gap: will print every gap number of batches
		val_num: number of examples for every test

		'''
		param = {'mode':'train','debug':self.debug,'reg':self.reg}
		if not val_num: val_num = self.batch_size

		train_acc, val_acc, train_loss, val_loss = [],[],[],[]

		num_batch = self.x_train.shape[0] / self.batch_size
		for ep in xrange(self.epoch):

			if verbose>0:
				print 'Epoch %d / %d:' % (ep + 1, self.epoch)

			for bt in xrange(num_batch):

				idx = np.random.choice(self.x_train.shape[0], self.batch_size)
				x_curr = self.x_train[idx]
				y_curr = self.y_train[idx]

				if self.debug:
					print 'input for layer 0:'
					print x_curr
					print '\n'

				for i in xrange(len(self.layers)):
					l = self.layers[i]
					x_curr = l.forward(x_curr,param)
					if bt==ep==0 and self.debug:
						print 'out from layer %d' % i
						print x_curr
						print '\n'
				# x_curr = prediction for current batch

				curr_acc = self.check_acc(x_curr, y_curr)
				curr_loss = self.layers[-1].getLoss(y_curr)
				
				for l in self.layers:
					w = l.get_kernel()
					if not w is None:
						curr_loss += 0.5 * self.reg * np.sum(w[:-1] * w[:-1])
						
				if self.debug:
					print 'curr_loss: %f' % curr_loss
				
				if bt % gap==0:
					train_acc.append(curr_acc)
					train_loss.append(curr_loss)

				if verbose>1 and bt % gap==0:
					
					print 'Pass %d / %d:' % (bt, num_batch)
					print 'Training accuracy: %f' % curr_acc
					print 'Training loss: %f' % curr_loss

				dldy = None
				for i in range(1,len(self.layers)+1):
					dldy = self.layers[-i].backward(dldy,param)
					if self.debug:
						print 'dldy for layer %d:' % (len(self.layers)-i)
						print dldy
					# So the update only works for constant lr,
					# what if we add changing lrs in the future?
					self.layers[-i].update(self.learning_rate)

				if bt % gap==0:
					curr_val_acc, curr_val_loss = self.test_accu_loss(val_num)
					val_acc.append(curr_val_acc)
					val_loss.append(curr_val_loss)
					if verbose>1:
						print 'Validation accuracy: %f' % curr_val_acc
						print 'Validation loss: %f' % curr_val_loss


		return train_acc, val_acc, train_loss, val_loss


	def __str__(self):
		# Used for debugging, easy way to print all relevant parameters
		s = '\n'

		s += 'x_train:\n' + str(self.x_train.shape) + '\n\n'
		s += 'y_train:\n' + str(self.y_train.shape) + '\n\n'
		s += 'x_val:\n' + str(self.x_val.shape) + '\n\n'
		s += 'y_val:\n' + str(self.y_val.shape) + '\n\n'

		s += 'learning rate:\n' + str(self.learning_rate) + '\n\n'
		s += 'batch size:\n' + str(self.batch_size) + '\n\n'
		s += 'epoch:\n' + str(self.epoch) + '\n\n'

		s += 'lr rule:\n' + str(self.lr_rule) + '\n\n'
		s += 'update rule:\n' + str(self.update_rule) + '\n\n'
		s += 'loss function:\n' + str(self.loss_func) + '\n\n'

		return s

if __name__=='__main__':
	pass

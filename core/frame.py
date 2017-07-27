import numpy as np

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
Loss Function:
	Softmax
Gradient Decent:
	SGD
	Adam (maybe)
'''

class Layer(object):
	'''
	The superclass for all layers in the frame.

	forward(self,x,param)
	backward(self,dldy,param)
	update(self,learning_rate)
	init_size(self,in)


	getKer(self)
	setKer(self)
	'''
	def forward(self, x, param):
		# Completes a forward pass and return y
		raise NotImplementedError("Function forward not implemented")

	def backward(self, dldy, param):
		# Completes a backward pass, update internal gradient and return dldx
		raise NotImplementedError("Function backward not implemented")

	def update(self, learning_rate):
		# Update kernel according to given learning rate
		raise NotImplementedError("Function update not implemented")

	def init_size(self, size):
		# Initialize internal parameters according to in (tuple) and 
		# return the out size
		raise NotImplementedError("Function init_size not implemented")

class lossCriteria(object):
	def getLoss(self):
		raise NotImplementedError("Function getLoss not implemented")

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
		'''
		self.layers = layers   # Last element is loss layer
		self.x_train = data['x_train']
		self.y_train = data['y_train']
		self.x_val = data['x_val']
		self.y_val = data['y_val']

		self.learning_rate = args.get('learning_rate',3e-3)
		self.batch_size = args.get('batch_size',50)
		self.epoch = args.get('epoch',10)
		
		# For now only does constant learning rate,  
		# sgd gradient update, softmax loss function,
		# maybe add more in the future?
		assert args.get('lr_rule','constant')=='constant',\
		'only accept constant for lr_rule'
		self.lr_rule = args.get('lr_rule','constant')
		
		assert args.get('update_rule','sgd')=='sgd',\
		'only accept sgd for update_rule'
		self.update_rule = args.get('update_rule','sgd')

		assert args.get('loss_func','softmax')=='softmax',\
		'only accept softmax for loss_func'
		self.loss_func = args.get('loss_func','softmax')

		size = (self.batch_size,) + self.x_train[0].shape
		print size ######## DEBUG ########
		for l in layers:
			size = l.init_size(size)

	def test_accu(self, num):
		'''
		Test the frame using x_val and y_val, check the accuracy
		of prediction
		'''
		param = {'mode':'test'}


	def train(self, verbose=0, gap=):
		'''
		Train the frame
		'''
		param = {'mode':'train'}

		num_batch = self.x_train.shape[0] / self.batch_size
		for ep in xrange(self.epoch):

			if verbose>0:
				print 'Epoch: ' + str(ep)

			for bt in xrange(num_batch):


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

	


















import numpy as np
import Layer as ly

class Softmax(ly.Layer, ly.LossCriteria):
	'''
	Softmax Loss layer, a bit different from the other layers in that
	it is the last layer of the frame and has an additional function
	calculating the loss of the frame.

	Parameters:
		x: input, (N, D)
		y: output/probability, (N, D)
		pred: prediction, (N, 1)
		loss: average scalar loss
		dldx: downgoing gradient (N, D)
		
	'''
	def __init___(self):
		self.x = None
		self.y = None
		self.pred = None
		self.loss = None
		self.dldx = None

	def init_size(self, size):
		self.x = np.zeros(size)
		self.dldx = np.zeros(size)
		self.pred = np.zeros((size[0], 1))
		self.y = np.zeros(size)
		self.loss = 0
		return (size[0], 1)

	def getLoss(self, y_true):
		'''
		Given the expected output of the network, compute the loss
		'''
		N, D = self.x.shape
		y_true = map(int, y_true)
		self.loss = -np.sum(np.log(self.y[np.arange(N), y_true])) / N
		self.dldx = self.y[:]
		self.dldx[np.arange(N), y_true] -= 1
		self.dldx /= N

		return self.loss

	def forward(self, x, param):
		'''
		Different from other forward functions. This on calculates
		the result/prediction of the whole network.
		'''
		assert x.shape==self.x.shape, \
		'Incompatible shape of x';

		self.x = x

		self.y = np.exp(x - np.max(x, axis=1, keepdims=True))
		self.y /= np.sum(self.y, axis=1, keepdims=True)

		self.pred = np.argmax(self.y, axis=1)

		return self.pred


	def backward(self, dldy, param):
		'''
		Calculates the gradient according to loss, x and y.
		Does not need dldy, just a courtesy for the 
		super class method.
		'''
		return self.dldx

	def update(self, learning_rate):
		pass

	def get_kernel(self):
		return None

	def __str__(self):
		s = ''

		s += '\nx:\n' + str(self.x) + '\n\n'
		s += '\ny:\n' + str(self.y) + '\n\n'
		s += '\nloss:\n' + str(self.loss) + '\n\n'
		s += '\ndldx:\n' + str(self.dldx) + '\n\n'
		s += '\npred:\n' + str(self.pred) + '\n\n'

		return s




import numpy as np
import Layer as ly

class Sigmoid(ly.Layer):
	'''
	Sigmoid layer: nonlinearity, takes in a 2D array

	Parameters:
		x: input,  (N,D)
		y: output, (N,D)
		dldy: ingoing gradient from upstream  (N,D)
		dldx: outgoing gradient to downstream (N,D)
	'''

	def __init__(self):
		self.x, self.y = None, None
		self.dldy, self.dldx = None, None


	def init_size(self,size):
		self.x = np.zeros(size)
		self.y = np.zeros(size)
		self.dldy = np.zeros(size)
		self.dldx = np.zeros(size)
		return size


	def forward(self, x, param):
		self.x = x
		size = x.shape
		self.y = 1.0 / (1 + np.exp(-self.x))
		return self.y


	def backward(self, dldy, param):
		assert dldy.shape==self.x.shape, 'Incompatible dldy size'
		self.dldy = dldy

		self.dldx = self.dldy * self.y * (1 - self.y)

		return self.dldx


	def update(self,learning_rate):
		pass

	def get_kernel(self):
		return None


	def __str__(self):
		s = ''

		s += '\nx:\n' + str(self.x) + '\n\n'
		s += '\ny:\n' + str(self.y) + '\n\n'
		s += '\ndldy:\n' + str(self.dldy) + '\n\n'
		s += '\ndldx:\n' + str(self.dldx) + '\n\n'

		return s

import numpy as np
import Layer as ly

class ReLu(ly.Layer):
	'''
	ReLu layer: nonlinearity, takes in a 2D array

	Parameters:
		aplha

		x: input,  (N,D)
		y: output, (N,D)
		dldy: ingoing gradient from upstream  (N,D)
		dldx: outgoing gradient to downstream (N,D)
	'''

	def __init__(self,alpha=0):
		self.alpha = alpha

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
		self.y = np.zeros(size)
		for i in xrange(size[0]):
			for j in xrange(size[1]):
				self.y[i,j] = max(x[i,j], x[i,j] * self.alpha)

		# print self.x
		# print self.y
		# print '\n'

		return self.y


	def backward(self, dldy, param):
		assert dldy.shape==self.x.shape, 'Incompatible dldy size'
		self.dldy = dldy

		s1, s2 = self.x.shape[0], self.x.shape[1]
		self.dldx = np.ones((s1, s2))
		for i in xrange(s1):
			for j in xrange(s2):
				if self.x[i,j]<0:
					self.dldx[i,j] = self.alpha * dldy[i,j]
				else:
					self.dldx[i,j] = dldy[i,j]
		return self.dldx



	def update(self,learning_rate):
		pass


	def __str__(self):
		s = ''

		s += '\nalpha:\n' + str(self.alpha) + '\n\n'
		s += '\nx:\n' + str(self.x) + '\n\n'
		s += '\ny:\n' + str(self.y) + '\n\n'
		s += '\ndldy:\n' + str(self.dldy) + '\n\n'
		s += '\ndldx:\n' + str(self.dldx) + '\n\n'

		return s


# Testing
if __name__=='__main__':
	l = ReLu(alpha=0.1)
	l.init_size((15,10))
	print l

	x = np.random.randn(15,10)
	l.forward(x,{})
	print l

	dldy = np.random.randn(15,10)
	l.backward(dldy,{})
	print l




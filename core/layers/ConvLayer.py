import numpy as np
import Layer as ly

class ConvLayer(ly.Layer):
	'''
	Convolutional Layer: learnable layer, takes in a tensor
	For now only allows 4D input

	Parameters:
		ft_num: number of filters
		ft_size: size of the filters
		stride: striding, gap between multiplication
		pad: padding, zero pad around result
		init_func: normal or random
		scale: initialization scale
		channel: number of (color) channels
		batch_size: number of data per run

		ker: kernel
		bias: numerical offset

	'''

	def __init__(self, ft_num, ft_size, stride, pad, \
		init_type='Random', scale=2e-2):

		'''
		ft_num: number of filters, #channel out
		ft_size: filter size
		stride: striding, gap between multiplication
		pad: padding, zero pad around result
		init_type: 'Normal', 'Random'
		scale: initialization scale
		'''

		assert init_type=='Normal' or init_type=='Random',\
		'init type needs to be Normal or Random'

		self.ft_num = ft_num
		self.ft_size = ft_size
		self.stride = stride
		self.pad = pad
		self.scale = scale
		self.init_func = np.random.randn if \
		init_type=='Normal' else np.random.rand

		self.in_wid = None
		self.in_hgt = None
		self.channel = None
		self.batch_size = None
		self.ker = None
		self.bias = None

		self.x = None
		self.y = None
		self.dldx = None
		self.dldy = None
		self.dk = None
		self.db = None



	def init_size(self,size):

		'''
		Assumes size[0] is batch size, size[-1] is # of channels
		'''

		assert len(size)==4, 'Only allows 4D input'
		assert (size[1] - self.ft_size) % self.stride==0
		assert (size[2] - self.ft_size) % self.stride==0

		self.in_hgt = (size[1] - self.ft_size) / self.stride + 1
		self.in_wid = (size[2] - self.ft_size) / self.stride + 1
		full_hgt = self.in_hgt + 2 * self.pad
		full_wid = self.in_wid + 2 * self.pad

		self.channel = size[-1]
		self.batch_size = size[0]
		self.ker = self.scale * self.init_func(\
			self.ft_num, self.ft_size, self.ft_size, self.channel)
		self.bias = self.scale * self.init_func(self.ft_num)

		self.x = np.zeros(size)
		self.y = np.zeros((size[0], full_hgt, full_wid, self.ft_num))

		return self.y.shape



	def forward(self, x, param):
		assert x.shape[0]==self.x.shape[0]

		self.x = x
		for f in xrange(self.ft_num):
			for n in xrange(self.batch_size):
				x_tmp = x[n].reshape(x.shape[1:])
				for h in xrange(self.in_hgt):
					for w in xrange(self.in_wid):
						curr = x_tmp[h * self.stride :h * self.stride + self.ft_size,\
							 		 w * self.stride :w * self.stride + self.ft_size]
						self.y[n, self.pad + h, self.pad + w, f] = \
						np.sum(curr * self.ker[f]) + self.bias[f]

		return self.y


	def backward(self, dldy, param):
		pass


	def update(self, learning_rate):
		pass


	def get_kernel(self):
		pass



if __name__=='__main__':
	l = ConvLayer(1,3,1,1)

	s = l.init_size((2,3,3,3))

	print s
	print "l.ker"
	print l.ker
	print "l.bias"
	print l.bias

	x = np.ones((2,3,3,3))
	print "x"
	print x
	y = l.forward(x,{})
	print "y"
	print y
	print np.sum(l.ker) + l.bias[0]


























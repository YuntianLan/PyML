import numpy as np
import Layer as ly
# Adapted code for fast vectorized calculation
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython

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
		self.ft_size = ft_size # Change of mind, this now is a tuple
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

		# Wierd stuff
		self.x_cols = None

	def init_size(self,size):

		'''
		Assumes size[0] is batch size, size[1] is # of channels
		'''

		assert len(size)==4, 'Only allows 4D input'
		assert (size[2] - self.ft_size[0]) % self.stride==0
		assert (size[3] - self.ft_size[1]) % self.stride==0

		self.in_hgt = (size[2] - self.ft_size[0]) / self.stride + 1
		self.in_wid = (size[3] - self.ft_size[1]) / self.stride + 1
		full_hgt = self.in_hgt + 2 * self.pad
		full_wid = self.in_wid + 2 * self.pad

		self.channel = size[1]
		self.batch_size = size[0]
		self.ker = self.scale * self.init_func(\
			self.ft_num, self.channel, self.ft_size[0], self.ft_size[1])
		self.bias = self.scale * self.init_func(self.ft_num)

		self.x = np.zeros(size)
		self.y = np.zeros((size[0], self.ft_num, full_hgt, full_wid))


		return self.y.shape



	def forward(self, x, param):
		assert x.shape[0]==self.x.shape[0]
		self.x = x

		b = self.bias
		w = self.ker
		N, C, H, W = x.shape

		num_filters = self.ft_num
		filter_height, filter_width = self.ft_size[0], self.ft_size[1]
		stride, pad = self.stride, self.pad

		# Check dimensions
		assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
		assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

		# Create output
		out_height = (H + 2 * pad - filter_height) / stride + 1
		out_width = (W + 2 * pad - filter_width) / stride + 1
		out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

		# x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
		x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
		self.x_cols = x_cols
		res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

		out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
		out = out.transpose(3, 0, 1, 2)

		self.y = out

		# cache = (x, w, b, conv_param, x_cols)
		# print 'output for conv forward: ' + str(out.shape)
		return out




		# This is fucking horrible
		# self.x = x
		# for f in xrange(self.ft_num):
		# 	for n in xrange(self.batch_size):
		# 		x_tmp = x[n].reshape(x.shape[1:])
		# 		for h in xrange(self.in_hgt):
		# 			for w in xrange(self.in_wid):
		# 				curr = x_tmp[h * self.stride :h * self.stride + self.ft_size,\
		# 					 		 w * self.stride :w * self.stride + self.ft_size]
		# 				self.y[n, self.pad + h, self.pad + w, f] = \
		# 				np.sum(curr * self.ker[f]) + self.bias[f]

		# return self.y


	def backward(self, dldy, param):
		x = self.x
		self.dldy = dldy
		w = self.ker
		b = self.bias
		dout = dldy
		x_cols = self.x_cols

		# x, w, b, conv_param, x_cols = cache
		stride, pad = self.stride, self.pad

		N, C, H, W = x.shape
		F, _, HH, WW = w.shape
		_, _, out_h, out_w = dout.shape

		db = np.sum(dout, axis=(0, 2, 3))

		dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
		dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

		dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
		dx_cols.shape = (C, HH, WW, N, out_h, out_w)
		dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

		self.dldx = dx
		self.dk = dw
		self.db = db

		return self.dldx


	def update(self, learning_rate):
		self.ker -= self.dk * learning_rate
		self.bias -= self.db * learning_rate


	def get_kernel(self):
		# Return None at this moment to avoid regularization
		return None


def conv_forward_im2col(x, w, b, conv_param):
	"""
	A fast implementation of the forward pass for a convolutional layer
	based on im2col and col2im.
	"""
	N, C, H, W = x.shape
	num_filters, _, filter_height, filter_width = w.shape
	stride, pad = conv_param['stride'], conv_param['pad']

	# Check dimensions
	assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
	assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

	# Create output
	out_height = (H + 2 * pad - filter_height) / stride + 1
	out_width = (W + 2 * pad - filter_width) / stride + 1
	out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

	# x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
	x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
	res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

	out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
	out = out.transpose(3, 0, 1, 2)

	cache = (x, w, b, conv_param, x_cols)
	return out, cache




if __name__=='__main__':
	l = ConvLayer(1,(2,2),1,0)

	s = l.init_size((2,3,3,3))

	x = l.x


	print s
	print "l.ker"
	print l.ker
	print "l.bias"
	print l.bias

	x = np.ones((2,3,3,3))
	print "x"
	print x
	y = l.forward(x, {})
	print "y"
	print y
	print np.sum(l.ker) + l.bias[0]

	dldx = l.backward(y, {})
	print "dldx"
	print dldx

























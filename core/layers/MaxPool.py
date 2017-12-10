import numpy as np
import Layer as ly
from fast_layers import *

from im2col import *
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython

import sys
sys.path.append('../..')
from data.data_util import *
sys.path.pop(-1)


# Functions adapted from Stanford CS231n for fast vectorized MaxPooling
def max_pool_forward_reshape(x, pool_param):
	"""
	A fast implementation of the forward pass for the max pooling layer that uses
	some clever reshaping.

	This can only be used for square pooling regions that tile the input.
	"""
	N, C, H, W = x.shape
	pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
	stride = pool_param['stride']
	assert pool_height == pool_width == stride, 'Invalid pool params'
	assert H % pool_height == 0
	assert W % pool_height == 0
	x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
	                     W / pool_width, pool_width)
	out = x_reshaped.max(axis=3).max(axis=4)

	cache = (x, x_reshaped, out)
	return out, cache


def max_pool_backward_reshape(dout, cache):
	"""
	A fast implementation of the backward pass for the max pooling layer that
	uses some clever broadcasting and reshaping.

	This can only be used if the forward pass was computed using
	max_pool_forward_reshape.

	NOTE: If there are multiple argmaxes, this method will assign gradient to
	ALL argmax elements of the input rather than picking one. In this case the
	gradient will actually be incorrect. However this is unlikely to occur in
	practice, so it shouldn't matter much. One possible solution is to split the
	upstream gradient equally among all argmax elements; this should result in a
	valid subgradient. You can make this happen by uncommenting the line below;
	however this results in a significant performance penalty (about 40% slower)
	and is unlikely to matter in practice so we don't do it.
	"""
	
	x, x_reshaped, out = cache

	dx_reshaped = np.zeros_like(x_reshaped)
	out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
	mask = (x_reshaped == out_newaxis)
	dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
	# print dout_newaxis.shape
	# print dx_reshaped.shape
	dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
	dx_reshaped[mask] = dout_broadcast[mask]
	dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
	dx = dx_reshaped.reshape(x.shape)

	return dx


def max_pool_forward_im2col(x, pool_param):
	"""
	An implementation of the forward pass for max pooling based on im2col.

	This isn't much faster than the naive version, so it should be avoided if
	possible.
	"""
	N, C, H, W = x.shape
	pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
	stride = pool_param['stride']

	assert (H - pool_height) % stride == 0, 'Invalid height'
	assert (W - pool_width) % stride == 0, 'Invalid width'

	out_height = (H - pool_height) / stride + 1
	out_width = (W - pool_width) / stride + 1

	x_split = x.reshape(N * C, 1, H, W)
	x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
	x_cols_argmax = np.argmax(x_cols, axis=0)
	x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
	out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

	cache = (x, x_cols, x_cols_argmax, pool_param)
	return out, cache


def max_pool_backward_im2col(dout, cache):
	"""
	An implementation of the backward pass for max pooling based on im2col.

	This isn't much faster than the naive version, so it should be avoided if
	possible.
	"""
	x, x_cols, x_cols_argmax, pool_param = cache
	N, C, H, W = x.shape
	pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
	stride = pool_param['stride']

	dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
	dx_cols = np.zeros_like(x_cols)
	dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
	dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
	          padding=0, stride=stride)
	dx = dx.reshape(x.shape)

	return dx





class MaxPool(object):
	'''
	Max Pooling Layer: Regularization layer, takes in a tensor
	For now only allow 4D input

	Parameters:
	size (tuple): dimension of pooling, length = 2
	stride (int): seperation between pooling
	max_idx (int): index of the maximum indices during pooling

	x(np.array): input
	y(np.array): output
	dldx(np.array): outgoing gradient to downstream
	dldy(np.array): ingoing gradient from upstream
	'''

	def __init__(self, size, stride):
		assert len(size)==2 and stride>0

		self.size = size
		self.stride = stride

		self.cache = None
		# self.max_idx = None

		self.x = None
		self.y = None
		self.dldx = None
		self.dldy = None


	def init_size(self, size):
		assert len(size)==4

		h, w = self.size[0], self.size[1]
		in_h, in_w = size[2], size[3]
		print self.size
		print size
		assert (in_h - h) % self.stride==0
		assert (in_w - w) % self.stride==0

		out_h = (in_h - h) / self.stride + 1
		out_w = (in_w - w) / self.stride + 1

		out_size = (size[0], size[1], out_h, out_w)
		self.x = np.zeros(size)
		self.y = np.zeros(out_size)
		self.max_idx = np.zeros(out_size, dtype=int)
		
		self.dldx = np.zeros(size)
		self.dldy = np.zeros(out_size)

		return out_size

	def forward(self, x, param):
		assert x.shape==self.x.shape

		self.x = x

		pool_param = {'pool_height':self.size[0], 
		'pool_width':self.size[1], 'stride':self.stride}

		# print 'x.shape:'
		# print self.x.shape
		# print '\nparameters:'
		# print pool_param


		N, C, H, W = x.shape
		pool_height, pool_width = self.size[0], self.size[1]
		stride = self.stride

		same_size = pool_height == pool_width == stride
		tiles = H % pool_height == 0 and W % pool_width == 0
		if same_size and tiles:
			out, reshape_cache = max_pool_forward_reshape(x, pool_param)
			cache = ('reshape', reshape_cache)
		else:
			out, im2col_cache = max_pool_forward_im2col(x, pool_param)
			cache = ('im2col', im2col_cache)

		self.cache = cache
		self.y = out
		return out

	def backward(self, dldy, param):
		method, real_cache = self.cache
		if method == 'reshape':
			self.dldx = max_pool_backward_reshape(dldy, real_cache)
			return self.dldx
		elif method == 'im2col':
			self.dldx = max_pool_backward_im2col(dldy, real_cache)
			return self.dldx
		else:
			raise ValueError('Unrecognized method "%s"' % method)

	def update(self, learning_rate):
		pass

	def get_kernel(self):
		return None

		# This, too, is stupid
		# ht, wt = self.y.shape[1], self.y.shape[2]
		
		# for n in xrange(x.shape[0]): # iterate each one
		# 	for h in xrange(ht):
		# 		for w in xrange(wt):
		# 			for c in xrange(x.shape[3]):
		# 				hs = h * self.stride
		# 				ws = w * self.stride
		# 				sub = x[n,hs:hs+self.size[0],ws:ws+self.size[1],c]
		# 				print '\n\nsub:'
		# 				print sub
		# 				print (n,h,w,c)
		# 				self.y[n,h,w,c] = np.max(sub)
		# 				self.max_idx[n,h,w,c] = np.argmax(sub)
		# return self.y

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__=='__main__':
	l = MaxPool((8,8),4)

	from scipy import misc
	import matplotlib.pyplot as plt
	# plt.figure()
	# img = rgb2gray(misc.face())[:,:,np.newaxis]
	# x = np.array([img.reshape((3,768,1024))])

	# print x.shape
	# print img.shape

	plt.imshow(misc.face())
	plt.show()

	x = np.array([misc.face().transpose()])
	l.init_size(x.shape)
	y = l.forward(x,{})[0]
	plt.imshow(y.transpose())
	plt.show()

	# # plt.imshow(img[:,:,0],cmap='gray')
	# # plt.show()
	# l.init_size(x.shape)
	# y = l.forward(x,{})[0,:,:,:]
	# plt.figure()
	# # y = y.reshape(y.shape[1], y.shape[2], y.shape[3])
	# print y.shape
	# y = y.reshape(192, 256, 3)
	# plt.imshow(y)
	# # plt.imshow(y[:,:,0],cmap='gray')
	# plt.show()




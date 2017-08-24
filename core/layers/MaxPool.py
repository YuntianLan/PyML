import numpy as np
import Layer as ly

class MaxPool(ly.Layer):
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

		self.max_idx = None

		self.x = None
		self.y = None
		self.dldx = None
		self.dldy = None


	def init_size(self, size):
		assert len(size)==4

		h, w = self.size[0], self.size[1]
		in_h, in_w = size[1], size[2]
		assert (in_h - h) % self.stride==0
		assert (in_w - w) % self.stride==0

		out_h = (in_h - h) / self.stride + 1
		out_w = (in_w - w) / self.stride + 1

		out_size = (size[0],out_h,out_w,size[3])
		self.x = np.zeros(size)
		self.y = np.zeros(out_size)
		self.max_idx = np.zeros(out_size, dtype=int)
		
		self.dldx = np.zeros(size)
		self.dldy = np.zeros(out_size)

		return out_size

	def forward(self, x, param):
		assert x.shape==self.x.shape

		self.x = x
		ht, wt = self.y.shape[1], self.y.shape[2]
		
		for n in xrange(x.shape[0]): # iterate each one
			for h in xrange(ht):
				for w in xrange(wt):
					for c in xrange(x.shape[3]):
						hs = h * self.stride
						ws = w * self.stride
						sub = x[n,hs:hs+self.size[0],ws:ws+self.size[1],c]
						print '\n\nsub:'
						print sub
						print (n,h,w,c)
						self.y[n,h,w,c] = np.max(sub)
						self.max_idx[n,h,w,c] = np.argmax(sub)
		return self.y
				
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__=='__main__':
	l = MaxPool((8,8),8)

	from scipy import misc
	import matplotlib.pyplot as plt
	plt.figure()	
	img = rgb2gray(misc.face())[:,:,np.newaxis]
	x = np.array([img])
	print x.shape
	print img.shape

	plt.imshow(img[:,:,0],cmap='gray')
	plt.show()
	l.init_size(x.shape)
	y = l.forward(x,{})[0,:,:,:]
	plt.figure()
	# y = y.reshape(y.shape[1], y.shape[2], y.shape[3])
	plt.imshow(y[:,:,0],cmap='gray')
	plt.show()




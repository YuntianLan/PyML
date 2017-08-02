import numpy as np
import Layer as ly

class DenseLayer(ly.Layer):
	'''
	Dense layer: learnable layer, takes in a 2D array

	Parameters:
		out_dim: output dimension
		init_func: function to initialize kernel
		init_type: 'Normal', 'Random'
		scale: 1
		w_size: shape of weights
	
		x: input  (N ,D1)   D1 = D + 1, x[:,D] = 1
		y: output (N, D2)
		w: weight (D1,D2)   w[D,:] = bias
		
		dldy: ingoing gradient from upstream
		dldx: outgoing gradient to downstream
		dw: kernel gradient


	TODO: should we support 3D and higher dimentional input?
	'''
	def __init__(self,out_dim,init_type='Normal',scale=2e-2):
		'''
		out_dim: number of dimensions each output vector has
		init_type: 'Normal', 'Random'
		'''

		assert type(out_dim)==int, 'Only support vector outputs'
		assert init_type=='Normal' or init_type=='Random',\
		'init type needs to be Normal or Random'

		self.out_dim = out_dim
		self.scale = scale
		self.init_type = init_type
		self.init_func = np.random.randn if init_type=='Normal'\
		else np.random.rand

		self.w_size = None
		self.x = None
		self.w = None
		self.dldy = None
		self.dldx = None
		self.dw = None


	def init_size(self,size):
		'''
		Initializes the kernel(self.w) based on input size.
		Returns the output size for the layer after.

		size: input size(tuple)
		'''
		bt_num = size[0]
		inter_size = size[1] + 1

		self.w_size = (inter_size,self.out_dim)
		self.w = self.init_func(inter_size,self.out_dim) * self.scale
		# self.w = 
		# array([[ 0.00508342,  0.00767761],
		#        [-0.00584968, -0.03521381],
		#        [-0.02020122,  0.00259685],
		#        [-0.00726217,  0.01616623],
		#        [-0.01109735, -0.0225348 ],
		#        [ 0.        ,  0.        ]])
		self.w[-1] = 0

		return (bt_num,self.out_dim)


	def forward(self, x, param):
		'''
		Forward pass for DenseLayer, append a column of 1's,
		multiply x with w and return the result(y)

		Needs x to be compatible with w's size
		TODO: what if sizes aren't compatible?
		'''
		x_size = x.shape
		x_size = (x_size[0], x_size[1] + 1)
		assert x_size[1]==self.w_size[0], 'Incompatible input shape'

		self.x = np.ones(x_size)
		self.x[:, :x_size[1] - 1] = x

		self.y = self.x.dot(self.w)

		# print self.x
		# print '\n'
		# print self.y
		# print '\n\n\n\n\n\n\n'

		return self.y


	def backward(self, dldy, param):
		# Backward pass for DenseLayer, calculates downstream
		# gradient and updates kernel gradient
		# TODO: do we need to consider the cases of test mode?
		assert dldy.shape[0]==self.x.shape[0] and \
			   dldy.shape[1]==self.out_dim,\
			   'Imcompatible dldy size'
		
		reg = param.get('reg',1e-3)
		self.dldy = dldy
		self.dldx = dldy.dot(self.w.T)
		
		self.dw = self.x.T.dot(dldy)
		temp = self.w[:]
		temp[-1] = 0
		self.dw += reg * temp

		if param.get('debug',0):
			print 'dw:'
			print self.dw
			print '\n'

		return self.dldx[:,:-1]


	def update(self,learning_rate):
		self.w -= self.dw * learning_rate


	def get_kernel(self):
		return self.w


	def __str__(self):
		s = ''

		s += '\nout_dim:\n' + str(self.out_dim) + '\n\n'
		s += '\ninit_type:\n' + self.init_type + '\n\n'
		s += '\nw_size:\n' + str(self.w_size) + '\n\n'

		s += '\nx:\n' + str(self.x) + '\n\n'
		s += '\nw:\n' + str(self.w) + '\n\n'
		s += '\ny:\n' + str(self.w) + '\n\n'

		s += '\ndldy:\n' + str(self.dldy) + '\n\n'
		s += '\ndldx:\n' + str(self.dldx) + '\n\n'
		s += '\ndw:\n' + str(self.dw) + '\n\n'

		return s

# Just for testing
if __name__=="__main__":
	l = DenseLayer(8)
	l.init_size((15,10))

	x = np.random.rand(15,10)
	dldy = np.random.rand(15,8)

	l.forward(x,{})
	l.backward(dldy,{})

	print l

	l.update(0.03)

	print l















import numpy as np
import Layer as ly

class SubtotalVectorMachine(ly.Layer, ly.LossCriteria):
	'''
	SVM Loss layer. It is another option for the network to calculate its loss.
	Parameters:
		init_func: function to initialize kernel
		init_type: 'Normal', 'Random'
		
		
		x: input, (N, C)
		y: output/probability, (N, C)
		
		VVVVVV# Jerry told me loss layer may or may not need weights so I'm getting rid of this matrix right now!
		w: weight (D,C)   w[D,:] = bias
		^^^^^^
		pred: prediction, (N, 1)
		loss: average scalar loss
		dldx: downgoing gradient (N, C)
		
		VVVVVVVVVVV No need for this param now!
		dldw:: updating kernel (D, C)
		^^^^^^^^^^^
		
	'''
	def __init__(self):
		######################################################################################
		##																					##
		##  I'm assuming out_dim is the number of classes in this data set. (C)				##
		##  Maybe it is not?																##
		##  Confused about what's y for this SVM layer, currently assuming to be the dot	##
		##  product between X and W. (X * W)												##
		##																					##
		######################################################################################
		######################################################################################
		##																					##
		##  Now these comments are useless													##
		##																					##
		######################################################################################
		'''
		assert type(out_dim)==int, 'Only support vector outputs'
		assert init_type=='Normal' or init_type=='Random',\
		'init type needs to be Normal or Random'
		
		self.out_dim = out_dim
		self.scale = scale
		self.init_type = init_type
		self.init_func = np.random.randn if init_type=='Normal'\
		else np.random.rand
		'''
		self.x = None
		self.y = None
		# self.w = None
		self.pred = None
		self.loss = None
		self.dldx = None
		# self.dldw = None

	def init_size(self, size):
		bt_num = size[0]
		inter_size = size[1] + 1
		
		self.x = np.zeros(size)
		'''
		self.w = self.init_func(inter_size,self.out_dim) * self.scale
		#                          ^ D + 1       ^ C
		'''
		self.dldx = np.zeros(size)
		self.pred = np.zeros((size[0], 1))
		self.y = np.zeros(size)
		self.loss = 0
		return size

	def getLoss(self, y_true):
		'''
		Given the expected output of the network, compute the loss
		'''
		N, D = self.x.shape
		correct_class_score = self.y[np.arange(N), y_true]
		correct_class_score_T = correct_class_score.reshape(N, 1)
		margins = self.y - correct_class_score_T + 1
		margins_zeroed = margins * (margins > 0)
		self.loss = np.sum(np.sum(margins_zeroed, axis = 1) - 1) / N
		
		dL = np.zeros(self.y.shape)
		dL[margins_zeroed > 0] = 1
		dL[range(N), list(y_true)] = 0
		dL[range(N), list(y_true)] = -np.sum(dL, axis = 1)
		#self.dldw = (self.x.T).dot(dL) / N

		self.dldx = dL / N
		#print str(self.dldx)
		#self.dldx = self.dldx[:, :self.dldx.shape[1] - 1]
		
		return self.loss

	def forward(self, x, param):
		'''
		Different from other forward functions. This one calculates
		the result/prediction of the whole network.
		'''
		self.x = x
		self.y = self.x

		# print 'forward forward forward forward'
		# print self.x
		# print self.y
		# print 'forward forward forward forward'

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
		# += '\nw:\n' + str(self.w) + '\n\n'
		s += '\ny:\n' + str(self.y) + '\n\n'
		s += '\nloss:\n' + str(self.loss) + '\n\n'
		s += '\ndldx:\n' + str(self.dldx) + '\n\n'
		#s += '\ndldw:\n' + str(self.dldw) + '\n\n'
		s += '\npred:\n' + str(self.pred) + '\n\n'

		return s
		
# Testing
if __name__=='__main__':
	l = SubtotalVectorMachine()

	l.init_size((2,3))
	print l

	y = l.forward(np.array([[2,3,4],[6,2,5]]),{})
	print y
	print l

	loss = l.getLoss(np.array([2,2]))
	print loss
	print l
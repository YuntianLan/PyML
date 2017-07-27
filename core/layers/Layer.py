
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

class LossCriteria(object):
	# Superclass for the loss layer
	def getLoss(self, y_true):
		raise NotImplementedError("Function getLoss not implemented")

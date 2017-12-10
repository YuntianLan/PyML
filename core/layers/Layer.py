
class Layer(object):
	'''
	The superclass for all layers in the frame.

	forward(self, x, param)
	backward(self, dldy, param)
	update(self, learning_rate)
	init_size(self, size)
	get_kernel(self)
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

	def get_kernel(self):
		# Return the kernel of the layer, None if there isn't any
		raise NotImplementedError("Function get_kernel not implemented")

class LossCriteria(object):
	# Superclass for the loss layer
	def getLoss(self, y_true):
		raise NotImplementedError("Function getLoss not implemented")

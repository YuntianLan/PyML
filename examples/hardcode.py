import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)

# Hardcoded 2 layer network to see what the heck is wrong
# w1: 784*100
# w2: 100:10

# x1 -> w1 -> x2

epoch, batch_size, learning_rate, reg = 1, 3, 3e-1, 1e-1
verbose = 1
gap = 30

# x, y = get_mnist_data('../data/mnist/mnist_train.csv',40000)

# x_t, y_t, x_v, y_v = x[:32000], y[:32000], x[32000:], y[32000:]
# data = {
# 	'x_train': x[:32000],
# 	'y_train': y[:32000],
# 	'x_val':   x[32000:],
# 	'y_val':   y[32000:]
# }

global w1, w2, dw1, dw2

w1 = np.random.randn(6,2) * 1
w1 = np.ones((6,2))
w1[-1] = 0
dw1 = np.zeros((6,2))

w2 = np.random.randn(3,2) * 1
w2 = np.ones((3,2))
w2[-1] = 0
dw2 = np.zeros((3,2))



def check_acc(y1, y2):
	assert y1.shape==y2.shape, 'Incompatible input'
	return len(filter(lambda x: abs(x)<1e-4, y1 - y2))/float(len(y1))


def forward(x_curr):

	global x1, x2, x3
	if x_curr is None:
		x_curr = np.array([
			[ 0.950802,0.162908,0.400773,0.895084,0.760840],
			[ 0.899969,0.917215,0.95278,0.632232, 0.445346],
			[ 0.283375,0.592097,0.921092,0.868487,0.049733]])
	x1 = np.ones((batch_size, 6))
	x1[:, :-1] = x_curr

	x2 = x1.dot(w1)
	x3_relu = x2 * (x2>0)
	x3 = np.ones((batch_size, 3))
	x3[:, :-1] = x3_relu

	x4 = x3.dot(w2)


	print 'x1:'
	print x1
	print '\n'

	print 'x2:'
	print x2
	print '\n'

	print 'x3:'
	print x3
	print '\n'

	print 'x4:'
	print x4
	print '\n'

	return x4

def lossCalc(x4,y_true):

	y = np.exp(x4 - np.max(x4, axis=1, keepdims=True))
	y /= np.sum(y, axis=1, keepdims=True)
	pred = np.argmax(y, axis=1)

	N, D = x4.shape
	print '###########'
	print y
	print y_true
	print '###########'
	loss = -np.sum(np.log(y[np.arange(N), y_true])) / N
	print 'loss1:' + str(loss)
	loss += 0.5 * reg * np.sum(w1[:-1] * w1[:-1])
	print w1
	print 'loss2:' + str(loss)
	loss += 0.5 * reg * np.sum(w2[:-1] * w2[:-1])
	print w2
	print 'loss3:' + str(loss)

	dout3 = y[:]
	dout3[np.arange(N), y_true] -= 1
	dout3 /= N

	print 'pred:'
	print pred
	print '\n'

	print 'dout3:'
	print dout3
	print '\n'

	return pred, dout3, loss


def backward(dout3):
	global dout2, dout1, dw1, dw2, w1, w2

	dout2 = dout3.dot(w2.T)
	dw2 = x3.T.dot(dout3)
	temp2 = w2[:]
	temp2[-1] = 0
	dw2 += reg * temp2

	dout1 = (x2>=0) * dout2[:,:-1]

	dw1 = x1.T.dot(dout1)
	temp1 = w1[:]
	temp1[-1] = 0
	# print dw1.shape
	# print temp1.shape
	dw1 += reg * temp1

	print 'dw2:'
	print dw2
	print '\n'

	print 'dw1:'
	print dw1
	print '\n'

	print 'dout2:'
	print dout2
	print '\n'

	print 'dout1:'
	print dout1
	print '\n'

	w1 -= learning_rate * dw1
	w2 -= learning_rate * dw2

t_acc, v_acc, t_loss, v_loss = [],[],[],[]

for epo in xrange(1,epoch+1):
	
	if verbose:
		print 'Starting of epoch %d / %d' % (epo, epoch)

	num_batch = 1 #len(x_t) / batch_size
	for bt in xrange(num_batch):
		#idx = np.random.choice(len(x_t),batch_size)
		x_curr, y_curr = None, np.array([0,1,0])#x_t[idx], y_t[idx]

		x4 = forward(None)
		pred, dout3, loss = lossCalc(x4, y_curr)
		t_acc.append(check_acc(pred, y_curr))
		t_loss.append(loss)

		backward(dout3)

		if bt % gap==0:
			#v_idx = np.random.choice(len(x_v),batch_size)
			#x_val, y_val = x_v[v_idx], y_v[v_idx]
			x_val = np.array([
				[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395],
				[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395],
				[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395],
				])
			y_val = np.array([0,0,0])
			x4_v = forward(x_val)
			# print type(x4_v)
			# print type(y_val)
			pred_v, dout3_v, loss_v = lossCalc(x4_v, y_val)
			v_acc.append(check_acc(pred_v, y_val))
			v_loss.append(loss_v)
			print 'Pass %d / %d' % (bt, num_batch)
			print 'Train accuracy: '+ str(t_acc[-1])
			print 'Train loss: '+ str(t_loss[-1])
			print 'Validation accuracy: '+ str(v_acc[-1])
			print 'Validation loss: '+ str(v_loss[-1])

l1 = len(t_acc)
l2 = len(v_acc)
print 'Average training accuracy: %f' % (sum(t_acc) / l1)
print 'Average Validation accuracy: %f' % (sum(v_acc) / l2)
print 'Average training loss: %f' % (sum(t_loss) / l1)
print 'Average validation loss: %f' % (sum(v_loss) / l2)
'''
plt.subplot(411)
plt.plot(range(len(t_loss)),t_loss)
plt.title('Training Loss')

plt.subplot(412)
plt.plot(range(len(t_acc)),t_acc)
plt.title('Train Accuracy')

plt.subplot(413)
plt.plot(range(len(v_loss)),v_loss)
plt.title('Validation Loss')

plt.subplot(414)
plt.plot(range(len(v_acc)),v_acc)
plt.title('Validation Accuracy')

plt.show()
'''









import numpy as numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)
import time


# A demonstration of using the frame to train a fully-connected
# network for cifar-10 dataset.
# Two hidden layers, 100 activations each
# Leaky ReLu nonlinearity
# Softmax cross entropy loss criteria
# Epoch: 5; Batch size: 60


layers = [
	DenseLayer(1000,scale=2e-2),
	ReLu(alpha=0.01),
	DenseLayer(500,scale=2e-2),
	ReLu(alpha=0.01),
	DenseLayer(250,scale=2e-2),
	ReLu(alpha=0.01),
	DenseLayer(10,scale=2e-2),
	Softmax()
]

x_t, y_t = np.zeros((50000,3072)), np.zeros(50000)
for i in xrange(5):
	x_sub, y_sub = get_cifar10_data('../data/cifar-10/data_batch_'+str(i+1),10000)
	x_t[10000*i : 10000*(i+1)] = x_sub.reshape(10000,3072)
	y_t[10000*i : 10000*(i+1)] = y_sub

x_v, y_v = get_cifar10_data('../data/cifar-10/test_batch',10000)
x_v = x_v.reshape(10000,3072)

x_t /= 64.; x_v /= 64.

data = {
	'x_train': x_t,
	'y_train': y_t,
	'x_val':   x_v,
	'y_val':   y_v
}


args = {'learning_rate':5e-3, 'epoch':3, 'batch_size':60, 'reg':1e-2, 'debug':0}

fm = frame(layers, data, args)

t1 = time.time()
train_acc, val_acc, train_loss, val_loss = fm.train(verbose=2,gap=10,val_num=500)
t2 = time.time()

print 'Time taken: %f' % (t2 - t1)

l = len(train_acc)
print 'Average training accuracy: %f' % (sum(train_acc) / l)
print 'Average Validation accuracy: %f' % (sum(val_acc) / l)
print 'Average training loss: %f' % (sum(train_loss) / l)
print 'Average validation loss: %f' % (sum(val_loss) / l)

print '\n'

print 'later half training accuracy: %f' % (sum(train_acc[l/2:]) / len(train_acc[l/2:]))
print 'later half Validation accuracy: %f' % (sum(val_acc[l/2:]) / len(val_acc[l/2:]))


plt.subplot(411)
plt.plot(range(len(train_loss)),train_loss)
plt.title('Training Loss')

plt.subplot(412)
plt.plot(range(len(train_acc)),train_acc)
plt.title('Train Accuracy')

plt.subplot(413)
plt.plot(range(len(val_loss)),val_loss)
plt.title('Validation Loss')

plt.subplot(414)
plt.plot(range(len(val_acc)),val_acc)
plt.title('Validation Accuracy')

plt.show()




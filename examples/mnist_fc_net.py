import numpy as numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)


# A demonstration of using the frame to train a fully-connected
# network for mnist dataset.
# Two hidden layers, 100 activations each
# ReLu nonlinearity
# Softmax cross entropy loss criteria
# Epoch: 5; Batch size: 50

layers = [
	DenseLayer(100,scale=2e-2),
	ReLu(alpha=0.01),
	DenseLayer(100,scale=2e-2),
	ReLu(alpha=0.01),
	DenseLayer(100,scale=2e-2),
	ReLu(alpha=0.01),
	Softmax()
]

x_t, y_t = get_mnist_data('../data/mnist/mnist_train.csv',50000)
x_v, y_v = get_mnist_data('../data/mnist/mnist_test.csv',10000)

data = {
	'x_train': x_t,
	'y_train': y_t,
	'x_val':   x_v,
	'y_val':   y_v
}


args = {'learning_rate':5e-3, 'epoch':2, 'batch_size':60, 'reg':1e-2, 'debug':0}

fm = frame(layers, data, args)

train_acc, val_acc, train_loss, val_loss = fm.train(verbose=2,gap=10,val_num=500)

l = len(train_acc)
print 'Average training accuracy: %f' % (sum(train_acc) / l)
print 'Average Validation accuracy: %f' % (sum(val_acc) / l)
print 'Average training loss: %f' % (sum(train_loss) / l)
print 'Average validation loss: %f' % (sum(val_loss) / l)

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




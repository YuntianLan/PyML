import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)

# A demonstration of using the frame to train a
# convolutional neural network for cifar-10 dataset.
# Conv -> ReLu -> Conv -> ReLu -> MaxPool ->
# Conv -> ReLu -> Conv -> ReLu -> MaxPool -> DenseLayer
# ReLu nonlinearity
# Softmax cross entropy loss criteria
# Epoch: 3; Batch size: 50


x_t, y_t = np.zeros((50000,3,32,32)), np.zeros(50000)
for i in xrange(5):
	x_t[10000*i : 10000*(i+1)], y_t[10000*i : 10000*(i+1)]=get_cifar10_data('../data/cifar-10/data_batch_'+str(i+1),10000)
x_v, y_v = get_cifar10_data('../data/cifar-10/test_batch',10000)
data = {
	'x_train': x_t,
	'y_train': y_t,
	'x_val':   x_v,
	'y_val':   y_v
}


args = {'learning_rate':1e-2, 'epoch':1, 'batch_size':100, 'reg':1e-2, 'debug':0}

layers = [
	ConvLayer(3,(5,5),1,0),
	ReLu(),
	ConvLayer(3,(5,5),1,4),
	ReLu(),
	MaxPool((8,8),8),
	ConvLayer(3,(5,5),1,4),
	ReLu(),
	ConvLayer(3,(5,5),1,4),
	ReLu(),
	MaxPool((2,2),2),
	ConvLayer(3,(5,5),1,4),
	ReLu(),
	ConvLayer(1,(5,5),1,4),
	ReLu(),
	MaxPool((2,2),2),
	DenseLayer(10, scale=1e-2),
	Softmax()
]



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











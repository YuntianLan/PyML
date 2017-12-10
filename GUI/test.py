import numpy as np
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)
import time

epoch = 3
learning_rate = 5e-3
batch_size = 60
reg = 1e-2


# Layers and data
layers = [
    DenseLayer(100,scale=2e-2),
    ReLu(alpha=0.01),
    DenseLayer(100,scale=2e-2),
    ReLu(alpha=0.01),
    DenseLayer(10,scale=2e-2),
    Softmax()
]
x_t, y_t = get_mnist_data('../data/mnist/mnist_train.csv',50000)
x_v, y_v = get_mnist_data('../data/mnist/mnist_test.csv',10000)

# x_t /= 100; x_v /= 100

# Let there be 500 points
num_run = 500
num_pass = len(x_t) * epoch / (batch_size * num_run)
print 'num_pass: ' + str(num_pass)

data = {
    'x_train': x_t,
    'y_train': y_t,
    'x_val':   x_v,
    'y_val':   y_v
}

# Build the network
args = {'learning_rate':learning_rate, 'epoch':epoch, 'batch_size':batch_size, 'reg':reg, 'debug':0}
fm = frame(layers, data, args)

# train_acc, val_acc, train_loss, val_loss = fm.train(verbose=2,gap=10,val_num=500)

print 'num_pass: ' + str(num_pass)
for i in xrange(num_run):
    train_acc, val_acc, _, _ = fm.passes(num_pass = num_pass, val_num = 10 * batch_size)
    print train_acc, val_acc
    # TODO: do whatever necessary with those 2 values









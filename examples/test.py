import numpy as numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)

# Just a test for the overfitting problem

layers = [
	DenseLayer(2,scale=1),
	ReLu(),
	DenseLayer(2,scale=1),
	Softmax()
]

x = np.array([
	[ 0.950802,0.162908,0.400773,0.895084,0.760840],
	[ 0.899969,0.917215,0.95278,0.632232, 0.445346],
	[ 0.283375,0.592097,0.921092,0.868487,0.049733]])
y = np.array([0,1,0])

x_t = np.array([[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395],
				[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395],
				[ 0.068066, 0.230542, 0.377966, 0.986269, 0.339395]])
y_t = np.array([0,0,0])

data = {
	'x_train': x,
	'y_train': y,
	'x_val':   x_t,
	'y_val':   y_t
}
args = {'epoch':1, 'batch_size':3, 'debug':1, 'reg':1e-1,
'learning_rate':3e-1}

fm = frame(layers, data, args)

train_acc, val_acc, train_loss, val_loss = fm.train(verbose=2,gap=50,val_num=1)


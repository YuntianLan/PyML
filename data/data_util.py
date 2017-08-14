import numpy as np
import cPickle as pickle
import csv
import os

cifar_10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


# Contains data getter functions

def get_mnist_data(name,num):
	'''
	Reads arbitary number of mnist data entries
	Returns:
	train_X: (num, 784), image for mnist dataset
	train_Y: (num, 1  ), corresponding labels
	'''

	with open(name,'rb') as train_file:
		train_reader = csv.reader(train_file,delimiter=',',quotechar='|')
		i = 0
		train_X = np.zeros((num + 2,784))
		train_Y = np.zeros(num + 2,dtype=int)
		for row in train_reader:
			i += 1
			if i>=num+2: break
			floatRow = map(float,row) # len = 785
			train_X[i] = np.array(floatRow[1:])
			train_Y[i] = np.array(int(floatRow[0]))
		train_Y = train_Y.T
	return train_X[2:], train_Y[2:]

def get_cifar10_data(name,num):
	'''
	Reads arbitary number of cifar-10 data entries
	Returns:
	X: (num, 32, 32, 3), image for cifar-10 dataset
	Y: (num, ), corresponding labels
	'''
	assert num<=10000, 'Number requested exceeded limit (10000)'
	
	try:
		f = open(name, 'rb')
	except IOError:
		# os.popen('sh cifar-10/source.sh')
		path = name[:name.rfind('/')]
		os.system('wget ' + path + ' ' + cifar_10_url)
		os.system('tar -xvf ' + path + '/cifar-10-python.tar.gz')
		os.system('rm ' + path + '/cifar-10-python.tar.gz')
		f = open(name, 'rb')

	datadict = pickle.load(f)
	X = datadict['data'][:num]
	Y = datadict['labels'][:num]
	X = X.reshape(num, 3, 32, 32).transpose(0,2,3,1).astype("float")
	Y = np.array(Y)
	f.close()
	return X, Y










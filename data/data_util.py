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
		print 'Unable to locate cifar10 data, downloading...'
		path = name[:name.rfind('/')]
		os.system('wget ' + path + ' ' + cifar_10_url)
		os.system('tar -xvf ' + path + '/cifar-10-python.tar.gz')
		os.system('rm ' + path + '/cifar-10-python.tar.gz')
		f = open(name, 'rb')
		print 'Donwload complete'

	datadict = pickle.load(f)
	X = datadict['data'][:num]
	Y = datadict['labels'][:num]
	X = X.reshape(num,3,32,32).astype("float") #transpose(0,2,3,1).
	Y = np.array(Y)
	f.close()
	return X, Y


'''
>>> x[0,:,:,0]
array([[  59.,   43.,   50., ...,  158.,  152.,  148.],
       [  16.,    0.,   18., ...,  123.,  119.,  122.],
       [  25.,   16.,   49., ...,  118.,  120.,  109.],
       ..., 
       [ 208.,  201.,  198., ...,  160.,   56.,   53.],
       [ 180.,  173.,  186., ...,  184.,   97.,   83.],
       [ 177.,  168.,  179., ...,  216.,  151.,  123.]])
>>> x[0,:,:,1]
array([[  62.,   46.,   48., ...,  132.,  125.,  124.],
       [  20.,    0.,    8., ...,   88.,   83.,   87.],
       [  24.,    7.,   27., ...,   84.,   84.,   73.],
       ..., 
       [ 170.,  153.,  161., ...,  133.,   31.,   34.],
       [ 139.,  123.,  144., ...,  148.,   62.,   53.],
       [ 144.,  129.,  142., ...,  184.,  118.,   92.]])
>>> x[0,:,:,2]
array([[  63.,   45.,   43., ...,  108.,  102.,  103.],
       [  20.,    0.,    0., ...,   55.,   50.,   57.],
       [  21.,    0.,    8., ...,   50.,   50.,   42.],
       ..., 
       [  96.,   34.,   26., ...,   70.,    7.,   20.],
       [  96.,   42.,   30., ...,   94.,   34.,   34.],
       [ 116.,   94.,   87., ...,  140.,   84.,   72.]])


'''







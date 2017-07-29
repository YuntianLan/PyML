import numpy as np
import csv

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

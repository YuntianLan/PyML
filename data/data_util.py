import numpy as np
import csv

# Contains data getter functions

def toFloat(lst):
	for i in range(len(lst)):
		lst[i] = float(lst[i])
	return lst

def get_mnist_data(num):
	'''
	Reads arbitary number of mnist data entries
	Returns:
	train_X: (num, 784), image for mnist dataset
	train_Y: (num, 1  ), corresponding labels
	'''
	with open('mnist/mnist_train.csv','rb') as train_file:
		train_reader = csv.reader(train_file,delimiter=',',quotechar='|')
		i = 0
		train_X = np.zeros((num,784))
		train_Y = np.zeros(num,dtype=int)
		for row in train_reader:
			i += 1
			if i>=num: break
			floatRow = toFloat(row) # len = 785
			train_X[i] = np.array(floatRow[1:])
			train_Y[i] = np.array(int(floatRow[0]))
		train_Y = train_Y.T
	return train_X, train_Y

from data_util import *
import numpy as np
import matplotlib.pyplot as plt

X, Y = get_cifar10_data('cifar-10/data_batch_1',10000)
X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8")

ig, axes1 = plt.subplots(5,5,figsize=(10,10))
for j in range(5):
	for k in range(5):
		i = np.random.choice(range(len(X)))
		axes1[j][k].set_axis_off()
		axes1[j][k].imshow(X[i:i+1][0])

plt.show()
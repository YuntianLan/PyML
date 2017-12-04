import numpy as numpy
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

layers = [
	
]
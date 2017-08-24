import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from core.frame import *
from data.data_util import *
sys.path.pop(-1)


x, y = get_cifar10_data('../data/cifar-10/data_batch_1',100)


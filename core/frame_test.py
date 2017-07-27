import frame as fm
import numpy as np

x1 = np.random.rand(10,5)
y1 = np.random.rand(10,1)
x2 = np.random.rand(2,5)
y2 = np.random.rand(2,1)

layers = []
data = {'x_train':x1, 'y_train':y1, 'x_val':x2, 'y_val':y2}

f = fm.frame(layers,data,{})

print f

import torch as th
from torch import nn
import numpy as np

size = 10
indices = np.arange(size)
sin = np.sin(6*indices/size) * np.ones((size, size))
cos = np.expand_dims(np.cos(6*indices/size), 1) * np.ones((size, size))
data = sin+cos
print(data[(0,0)])
        
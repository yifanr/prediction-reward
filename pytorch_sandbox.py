import torch as th
from torch import nn

lstm = nn.LSTM(4, 8, num_layers=1,)
input = th.randn(1,5,4)
print(input.shape)
print(lstm.input_size)
output, states = lstm(input)
print(states[0])
print(states[1])
print(output)


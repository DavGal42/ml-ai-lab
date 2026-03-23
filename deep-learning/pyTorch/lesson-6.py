import torch
import torch.nn as nn

# model = nn.Sequential()
# model.add_module('layer_1', nn.Linear(784, 128))
# model.add_module('relu', nn.ReLU())
# model.add_module('layer_2', nn.Linear(128, 10))

# print(model)
# print(model.layer_1)
# print(model.relu)
# print(model.layer_2)

# input = torch.randn(1, 784, dtype=torch.float32)
# output = model(input)
# print(output.shape)


# -------------------------------------------

class MyModel(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer_1 = nn.Linear(input, 128)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(128, output)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        out = self.layer_2(x)
        return out
    

model = MyModel(784, 10)
# print(model)
# print(model.layer_1)
# print(model.relu)
# print(model.layer_2)

input = torch.randn(1, 784, dtype=torch.float32)
output = model(input)
# print(output.shape)
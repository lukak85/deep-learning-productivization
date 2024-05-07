import torch
import torch.nn as nn
import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define model layers here

    def forward(self, x):
        # Define forward pass here
        return x


model = MyModel()

torch.save(model.state_dict(), "sloberta-squad2-SLO.pth")

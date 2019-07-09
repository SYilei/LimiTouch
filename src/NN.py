import torch
import torch.nn.functional as F
import random as rd
import numpy as np
import time
from os import listdir
import PrepareData as pd


################## Classes #################
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)    # hidden layer1
        self.predict = torch.nn.Linear(n_hidden, n_output)    # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))                            # activation function for hidden layer
        x = self.predict(x)                                   # linear output
        return x
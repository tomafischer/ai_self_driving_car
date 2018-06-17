# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#Create the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
        
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        #event: 1) last state st 2) next state st+1 
        #       3)last action at 4) reward rd
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        

##signal
#last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
# with signal1 being float
#last reward = -1 opr 1 or 0.2 (float value)
#rotation float value
# e.g. last signal = [1,1,1,45,-45]            
            
            
#### extra part for understanding
m1= [(1,2,3,4),(2,3,4,5),(3,4,5,6),(4,5,6,7,8)]
samples = zip(*random.sample(m1, 4))
list(samples)
#-> Out[107]: [(3, 1, 2, 4), (4, 2, 3, 5), (5, 3, 4, 6), (6, 4, 5, 7)]
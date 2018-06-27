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
        
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        #event: 1) last state st 2) next state st+1 
        #       3)last action at 4) reward rd
        # e.g. state  1 (sensor1) 1 (sensor2) 1 (sensor3) 45 (orientation) -45
        #      last action 0  - reward = 1    
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    def samples(self, batch_size):
        #list = ((1,2,3),(4,5,6)) => zip(*list) = (1,4), (2,5), (3,6)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x,0), samples)


    
        
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []

        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr= 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        T = 100
        # state.requires_grad = False
        with torch.no_grad():
            probs =  F.softmax(self.model(state ) * T, dim=1)
            action = probs.multinomial(num_samples=1)
        return action.data[0,0]
     
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        '''The hard of the deep q-learning. Implements the formular
        
        '''
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # outputs will be the q value of the state we took based on the action e.g. 19
        # Q(s,a)
        # brain.model(st0) -> tensor([[-19.3112, -13.8050, -13.2917]])
        # action -> 0
        # -> tensor([-19.3112])
        
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #detach returns a new Tensor, detached from the current graph.
        # -> brain.model(st1) -> tensor([[ 3.5216, -4.2244, -4.6560]])
        # -> tensor([ 3.5216])
        
        # page 7: target = reward + gamma * max(Q(a, s1))
        target = self.gamma*next_outputs + batch_reward
       

        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
      
    def update(self, reward, new_signal):
        #convert new_signal to tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #push to memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),torch.Tensor([reward])))
        #action  = take an action
        action = self.select_action(new_state)
        #check if we have to train our network
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.samples(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #update action, state , reward
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        #appending reward and making sure it is not more than 1000
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.0)
    
    def save(self, filename ='last_brain.pth'):
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimzer': self.optimizer.state_dict()}, f = filename)
        print('Saved model as: {} '.format(filename))
        
    def load(self, filename ='last_brain.pth'):
        if os.path.isfile(filename):
            print("Loading the model: {}".format(filename))
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimzer'])
        else:
            print("No model found: {}".format(filename))
        
        ##signal
#last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
# with signal1 being float
#last reward = -1 opr 1 or 0.2 (float value)
#rotation float value
# e.g. last signal = [1,1,1,45,-45] 
            
#understandig the sample method in Replay Memory
if __name__ == '__main__':
    st0 = torch.Tensor([1,1,1,45,-45]).float().unsqueeze(0)
    st1 = torch.Tensor([0,1,1,45,-45]).float().unsqueeze(0) 
    action = torch.LongTensor([int(0)])
    reward = torch.Tensor([0.2])
    replayMem = ReplayMemory(20)
    for i in range(20):
        replayMem.push((st0, st1, action, reward))
    samples = zip(*random.sample(replayMem.memory, 5)) 
    batch = map(lambda x: torch.cat(x, 0), samples) 
    list(batch)
    list(replayMem.samples(2)) 
    #### extra part for understanding
    m1= [(1,2,3,4),(2,3,4,5),(3,4,5,6),(4,5,6,7,8)]
    samples = zip(*random.sample(m1, 4))
    lsamples = list(samples)
    batch = map(lambda x: torch.cat(x,0), lsamples)
    #-> Out[107]: [(3, 1, 2, 4), (4, 2, 3, 5), (5, 3, 4, 6), (6, 4, 5, 7)]
    
    #####testing Dqn
    brain = Dqn(5,3,0.9)
    output = brain.model(st0)
    probs = F.softmax(output, dim = 1)
    print("output: {}".format(output))
    print("props: {}".format(probs))
    action = probs.multinomial(num_samples=1)
    print("action: {}".format(action.data[0,0]))
    
    print("brain_select: {}".format(brain.select_action(st0)))
    
    ###testing learn
    st0 = torch.Tensor([1,1,1,45,-45]).float().unsqueeze(0)
    action = torch.LongTensor([int(0)])
    q0 =brain.model(st0).gather(1,action.unsqueeze(1)).squeeze(1)
    print("q0: {}".format(q0))
    # brain.model(st0) -> tensor([[-19.3112, -13.8050, -13.2917]])
    # action -> 0
    # -> tensor([-19.3112])
    maxQ1 =brain.model(st1).detach().max(1)[0]
    # -> brain.model(st1) -> tensor([[ 3.5216, -4.2244, -4.6560]])
    # -> tensor([ 3.5216])
    print("max(q1): {}".format(maxQ1))
    
    
    
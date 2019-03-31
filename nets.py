import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

###########################################################################
#
#                               Classes
#
###########################################################################

class Memory:
    '''
    Description:
    The Memory class allows to store and sample events

    Attributes:  
    capacity -- max amount of events stored
    data -- list with events memorized
    pointer -- position of the list in which an event will be registered

    Methods:
    store -- save one event in "data" in the position indicated by "pointer"
    sample -- returns a uniformly sampled batch of stored events
    retrieve -- returns the whole information memorized
    forget -- elliminates all data stored
    '''

    def __init__(self, capacity = 50000):
        '''
        Description:
        Initializes an empty data list and a pointer located at 0. 
        Also determines the capacity of the data list.

        Arguments:
        capacity -- positive int number
        '''
        self.capacity = capacity
        self.data = []        
        self.pointer = 0

    
    def store(self, event):
        '''
        Description:
        Stores the input event in the location designated by the pointer.
        The pointer is increased by one modulo the capacity. 

        Arguments:
        event -- tuple to be stored
        '''
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.capacity

    
    def sample(self, batch_size):
        '''
        Description:
        Samples a specified number of events 

        Arguments:
        batch_size -- int number that determines the amount of events to be sampled

        Outputs:
        random list with stored events
        '''
        return random.sample(self.data, batch_size) 

    def retrieve(self):
        '''
        Description:
        Returns the whole stored data  

        Outputs:
        data
        '''
        return(self.data)
    
    def forget(self):
        '''
        Description:
        Restarts the stored data and the pointer
        '''
        self.data = []
        self.pointer = 0

#-------------------------------------------------------------
#
#    Value network
#
#-------------------------------------------------------------
class v_valueNet(nn.Module):
    '''
    Description:
    The valueNet is a standard fully connected NN with ReLU activation functions
    and 3 linear layers that approximates the value function

    Attributes:  
    l1,l2,l3 -- linear layers
    
    Methods:
    forward -- calculates otput of network
    '''
    def __init__(self, input_dim):
        '''
        Description:
        Creates the three linear layers of the net

        Arguments:  
        input_dim -- int that specifies the size of input        
        '''
        super().__init__()        
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)  

        self.l3.weight.data.uniform_(-3e-3, 3e-3)
        self.l3.bias.data.uniform_(-3e-3, 3e-3)       

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)    
    
    def forward(self, s):
        '''
        Descrption:
        Calculates output for the given input

        Arguments:  
        x -- input to be propagated through the net

        Outputs:
        x -- number that represents the approximate value of the input        
        '''
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

class q_valueNet(nn.Module):
    '''
    Description:
    The valueNet is a standard fully connected NN with ReLU activation functions
    and 3 linear layers that approximates the value function

    Attributes:  
    l1,l2,l3 -- linear layers
    
    Methods:
    forward -- calculates otput of network
    '''
    def __init__(self, s_dim, a_dim):
        '''
        Descrption:
        Creates the three linear layers of the net

        Arguments:  
        input_dim -- int that specifies the size of input        
        '''
        super().__init__()        
        self.l1 = nn.Linear(s_dim+a_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)  

        self.l3.weight.data.uniform_(-3e-3, 3e-3)
        self.l3.bias.data.uniform_(-3e-3, 3e-3) 

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)    
    
    def forward(self, s,a):
        '''
        Descrption:
        Calculates output for the given input

        Arguments:  
        x -- input to be propagated through the net

        Outputs:
        x -- number that represents the approximate value of the input        
        '''
        x = torch.cat([s, a], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return(x)

#-------------------------------------------------------------
#
#    Policy network
#
#-------------------------------------------------------------
class policyNet(nn.Module):
    '''
    Description:
    The policyNet is a standard fully connected NN with ReLU and sigmoid activation 
    functions and 3 linear layers. This net determines the action for a given state. 

    Attributes:  
    l1,l2,l3 -- linear layers
    
    Methods:
    forward -- calculates otput of network
    '''
    def __init__(self, input_dim, output_dim, min_log_stdev=-30, max_log_stdev=30):
        '''
        Descrption:
        Creates the three linear layers of the net

        Arguments:  
        input_dim -- int that specifies the size of input        
        '''
        super().__init__()     
        self.min_log_stdev = min_log_stdev
        self.max_log_stdev = max_log_stdev

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l31 = nn.Linear(256, output_dim)
        self.l32 = nn.Linear(256, output_dim)

        self.l31.weight.data.uniform_(-3e-3, 3e-3)
        self.l32.weight.data.uniform_(-3e-3, 3e-3)
        self.l31.bias.data.uniform_(-3e-3, 3e-3)
        self.l32.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)    
    
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        m = self.l31(x)
        log_stdev = self.l32(x)
        log_stdev = torch.clamp(log_stdev, self.min_log_stdev, self.max_log_stdev)
        return m, log_stdev
    
    def sample_action(self, s):
        '''
        Description:
        Calculates output for the given input

        Arguments:  
        x -- input to be propagated through the net

        Outputs:
        a --         
        '''
        m, log_stdev = self(s)
        u = m + log_stdev.exp()*torch.randn_like(m)
        a = torch.tanh(u).cpu()        
        return a

    def sample_action_and_llhood(self, s):
        m, log_stdev = self(s)
        stdev = log_stdev.exp()
        u = m + stdev*torch.randn_like(m)
        a = torch.tanh(u)
        llhood = (Normal(m, stdev).log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), 1e-6, 1.0))).sum(dim=1, keepdim=True)
        return a, llhood
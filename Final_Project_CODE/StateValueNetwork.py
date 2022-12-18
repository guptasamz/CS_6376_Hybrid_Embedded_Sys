import torch.nn as nn
import torch.nn.functional as F

#Using a neural network to learn state value
class StateValueNetwork(nn.Module):
    
    
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer= nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        
    def forward(self, x):
  
        x = self.input_layer(x)
      
        activation1 = F.relu(x)
        h= self.hidden_layer(activation1)
        activation2=F.relu(h)
      
        state_value = self.output_layer(activation2)
        
        return state_value
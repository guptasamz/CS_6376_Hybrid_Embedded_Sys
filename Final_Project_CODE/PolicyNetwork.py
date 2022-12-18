import torch.nn as nn
import torch.nn.functional as F

#Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
    
    #Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.hidden_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_space)
    
    #forward pass
    def forward(self, x):
        
        x = self.input_layer(x)
        
       
        activation1 = F.relu(x)
        h= self.hidden_layer(activation1)
        activation2= F.relu(h)
        
        actions = self.output_layer(activation2)
        
        #get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)
        
        return action_probs
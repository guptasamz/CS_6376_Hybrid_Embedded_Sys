import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
from tqdm import tqdm
import numpy as np
from collections import deque

from PolicyNetwork import PolicyNetwork


#Using a neural network to learn our policy parameters
class CartPoleTest():
    
    #Takes in observations and outputs actions
    def __init__(self,env, state):
        self.current_state = state
        self.env=env
        self.policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        
    
    #forward pass
    def select_action(self):
        state=self.current_state
        state = torch.from_numpy(state).float().unsqueeze(0)

        # use network to predict action probabilities
        action_probs = self.policy_network(state)
        state = state.detach()

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample().item()

        # return action
        return action
    
    def step_action(self, action):
        new_state, reward, done, info,_ = self.env.step(action)
        #select action based on new state
        self.current_state=new_state
        action=self.select_action() 
        return new_state, reward, done,action
    
    def load_weights(self,weights):
        self.policy_network.load_state_dict(weights)
        
        
   
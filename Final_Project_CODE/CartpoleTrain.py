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
from StateValueNetwork import StateValueNetwork
#Using a neural network to learn our policy parameters
class CartPoleTrain():
    def __init__(self,env, state):
        self.current_state = state
        self.env=env
        self.policy_network = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.stateval_network = StateValueNetwork(env.observation_space.shape[0])
        #Init optimizer
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-2)
        self.stateval_optimizer = optim.Adam(self.stateval_network.parameters(), lr=1e-2)
        self.gamma=0.99
        self.num_episodes=10000
        self.max_training_steps=10000
        self.solved_score=1000
        self.scores = []
        
        #Generates an action and its log Probability from policy action
    def select_action(self):
     
        state=self.current_state
        state = torch.from_numpy(state).float().unsqueeze(0)

        action_probs = self.policy_network(state)
        state = state.detach()

        m = Categorical(action_probs)
        action = m.sample()
        log_prob=m.log_prob(action)
        action=action.item()
        return action, log_prob

        #Estimate normalized Returns of a trajectory
    def process_rewards(self,rewards):

        total_r = 0
        G = []

        for r in reversed(rewards):

            total_r =  total_r * self.gamma + r 


            G.insert(0, total_r)
        G = torch.tensor(G)
        return (G - G.mean()) / G.std()

        #Updates Policy Network weights using policy loss calculated by using (Returns-Predicted state values) and log probablities
    def train_policy(self,deltas, log_probs, optimizer):
     

        policy_loss_list = []

        for d, lp in zip(deltas, log_probs):
            
            policy_loss_list.append(-d * lp)
        
        policy_loss=sum(policy_loss_list)
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        #Updates State Value Network weights using MSE Loss calculated by using Returns-Predicted state values
    def train_value(self,G, state_vals, optimizer):

        val_loss = torch.nn.MSELoss()(state_vals, G)

        optimizer.zero_grad()
        val_loss.backward()
        optimizer.step()
        
    def learn(self):
        
        
        for episode in tqdm(range(self.num_episodes)):
            
            
            if(episode!=0):
                self.current_state=self.env.reset()[0]

            score = 0
            states = []

            rewards = []
            lps = []
            
            #Generate a Trajectory
            for step in range(self.max_training_steps):

                action, lp = self.select_action()
                

                new_state, reward, done, _,_ = self.env.step(action)
                

                score += reward
                

                states.append(self.current_state) 
                rewards.append(reward)
                lps.append(lp)
                
                if done:
                    break
                
                self.current_state = new_state
            
            self.scores.append(score)
            
            
            if np.array(self.scores[-50:]).mean() >= self.solved_score and len(self.scores) >= 50:
                break
            

            #Estimate Returns for the trajectory
            G = self.process_rewards(rewards)
        
            #Estimate the gradient for State Value Network 
            state_vals = []
            for state in states:
                state = torch.from_numpy(state).float().unsqueeze(0)
                state_vals.append(self.stateval_network(state))
                
            state_vals = torch.stack(state_vals).squeeze()
            
            #Update the weights of State Value Network
            self.train_value(G, state_vals, self.stateval_optimizer)
             
            #Estimate the gradient for Policy Network and update weights
            deltas = [gt - val for gt, val in zip(G, state_vals)]
            deltas = torch.tensor(deltas)
            self.train_policy(deltas, lps, self.policy_optimizer)
            
        self.env.close()
        
        return self.policy_network.state_dict()
    
    #Plot training scores
    def plot(self):
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import seaborn as sns
        import numpy as np

        sns.set()

        plt.plot(self.scores)
        plt.ylabel('score')
        plt.xlabel('episodes')
        plt.title('Training score of CartPole')

        reg = LinearRegression().fit(np.arange(len(self.scores)).reshape(-1, 1), np.array(self.scores).reshape(-1, 1))
        y_pred = reg.predict(np.arange(len(self.scores)).reshape(-1, 1))
        plt.plot(y_pred)
        plt.savefig('Training_Scores.png')
                
            

                
        
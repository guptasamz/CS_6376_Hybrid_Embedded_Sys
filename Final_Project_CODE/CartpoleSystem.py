import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym
from tqdm import tqdm
import numpy as np
from collections import deque
from CartpoleTrain import CartPoleTrain
from CartpoleTest import CartPoleTest
from PolicyNetwork import PolicyNetwork


done = False
env = gym.make('CartPole-v1')

scores = []

state = env.reset(seed=115)[0]
done = False
score = 0
i=0
trained_state=False
action=None
test=None

# Each loop is a clock cyle
while not done:
        
        #Component1:Trainer
        #Input: Initial Conditions
        #State Variables: weights of policy and state value Networks, trained_state
        #Output: Policy Network Weights
        if not trained_state:
            train=CartPoleTrain(env=env, state=state)
            trainer_state_dicts=train.learn()
            trained_state=True
            train.plot()

        
        #Component2: Testet
        #Input: Initial Conditions, Network Weights from Trainer, Current Action 
        #State Variables: weights of policy Network, current conditions(state), done
        #Output: Next Action, Reward
        if test is None:
            env=gym.make('CartPole-v1',render_mode='human')
            state=env.reset(seed=115)[0]
            test=CartPoleTest(env=env,state=state)
        else:
            test.load_weights(trainer_state_dicts)
        
        if action is not None:
            new_state, reward, done,action=test.step_action(action=action)
            score += reward
            state = new_state
        else:
            action=test.select_action()
        
       
        i+=1
        print("Clock Cycle: "+str(i))
        
scores.append(score)
#This indicates that the component has failed and it has to be reset
print( "Error State entered")

print(np.array(scores).mean())


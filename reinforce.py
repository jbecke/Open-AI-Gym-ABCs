import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random


env = gym.make('MountainCar-v0').env
env.seed(0)
torch.manual_seed(0)
discount = 0.99
epsilon = 0.3

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    # Take action according to probability of the outputs
    if (random.random() < epsilon): # with probability epsilon
        action = torch.tensor([random.randint(0, len(probs) - 1)])
    else: 
        action = m.sample()
    # Take the log of the probability of the selected action
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item() # convert to int


def train():
    R = 0
    policy_loss = []
    rewards = []
    # Compute discounted rewards
    for r in policy.rewards[::-1]:
        R = r + discount * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # normalize and dont div by 0

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad() # Zeroes out the previous gradients
    # Similar to torch.stack
    # torch.stack converts [tensor(1), tensor(2), tensor(3)] to tensor([ 1,  2,  3]) and sum converts to tensor(6)
    policy_loss = torch.cat(policy_loss).mean() # The mean reward over the whole episode
    print(policy_loss)
    policy_loss.backward() # Compute gradients over every Variable in the network
    optimizer.step() # Apply the computed gradients to every Variable in the network
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            #print(state[0])
            if i_episode % 10 == 0:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        train() # train the model

        # Print diagnostics
       
        print('Episode {}\tLast length: {:5d}\t'.format(
            i_episode, t))


if __name__ == '__main__':
    main()

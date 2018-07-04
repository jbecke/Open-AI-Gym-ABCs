import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

GAMMA = 0.99
env = gym.make('MountainCar-v0').env
env.seed(0)
torch.manual_seed(0)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(2, 1000)
        # Action head deterins distributions over the actions, i.e. the actor
        self.action_head = nn.Linear(1000, 3)
        # Value head predicts the value for the current state, i.e. the critic
        self.value_head = nn.Linear(1000, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def train():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    # Compute discounted future rewards for each state by iterating backwards through immediate rewards
    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    # Normalization step
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # add eps to not divide by 0
    # value is the critic's prediction for the value of the state
    for (log_prob, value), r in zip(saved_actions, rewards):
        # adjust the actor reward the reflect the prior expectation
        # i.e. if the discounted reward from the state was equal to the expected reward (i.e. the value), then don't update the policy
        # if the reward was greater than the expected value (e.g. -20 instead of the expected -100), give a positive reward to increase
        # the likliness of that state
        # conversely, we want to decrease the probability if the actual reward was less than the expected reward
        actor_reward = r - value.item() 
        policy_losses.append(-log_prob * actor_reward) # for training the actor

        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r]))) # for training the critic
    for param in model.value_head.parameters():
        if param.requires_grad:
            print(param.grad)
    optimizer.zero_grad()
    # torch.stack converts [tensor(1), tensor(2), tensor(3)] to tensor([ 1,  2,  3]) and sum converts to tensor(6)
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward() 
    # How does loss 
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
 

def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if i_episode % 1000 == 0:
                env.render() 
            model.rewards.append(reward)
            if done:
                if reward >= 198:
                    continue
                else:
                    break

        running_reward = running_reward * 0.99 + t * 0.01 # Just for viewing progress purposes
        
        train() # train the model

        # Print diagnostics
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))


if __name__ == '__main__':
    main()
# Open AI Gym ABCs

Many RL tutorials contain superfluous code or use a Jupyter notebook with lots of text. This repo aims to be a no-frills  implementation (i.e. maximally shortened) of REINFORCE and Actor Critic RL methods.

## Suggestions

Start by reading over reinforce.py as it is the simplest method. After that, read about actor critic methods ([see this comic](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)), then look at the code. The comments assume basic knowledge of RL. It helps to have an understanding of traditional RL (i.e. pre-NNs, [Bellman equations](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/)) before trying your hand with Pytorch or OpenAI Gym.

## Prereqs

* OpenAI gym with the cartpole environment installed
* Python 3.6 (but older versions including Python 2 may work)
* numpy

## Training time 

The models shold train within several minutes on a modern laptop CPU.

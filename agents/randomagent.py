import random
import numpy as np

from .base import Agent

class RandomAgent(Agent):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self, observation):
        action_space = [i for i in range(self.n_actions) if np.isclose(observation[i], -1)]
        if len(action_space) == 0:
            return None
        return random.choice(action_space)

    def learn(self, *args, **kwargs):
        return
        
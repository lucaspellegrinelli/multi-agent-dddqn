import random
import numpy as np

from .base import Agent
from envirorment import Snake

class SnakeRandomAgent(Agent):
    def _hit_something(self, obs_board, position):
        return position[0] < 0 or position[0] > 9 or position[1] < 0 or position[1] > 9 or \
            obs_board[position[0], position[1]] == Snake.AGENT_B_HEAD_ID or \
            obs_board[position[0], position[1]] == Snake.AGENT_B_BODY_ID or \
            obs_board[position[0], position[1]] == Snake.AGENT_A_BODY_ID or \
            obs_board[position[0], position[1]] == Snake.AGENT_A_HEAD_ID
            
    def act(self, observation):
        obs_board = np.reshape(observation, (10, 10))

        # find where agent a head is
        agent_a_head = np.where(obs_board == Snake.AGENT_A_HEAD_ID)
        agent_a_head = (agent_a_head[0][0], agent_a_head[1][0])

        # check directions where agent_a can move
        valid_actions = []
        up_will_hit = self._hit_something(obs_board, (agent_a_head[0] - 1, agent_a_head[1]))
        down_will_hit = self._hit_something(obs_board, (agent_a_head[0] + 1, agent_a_head[1]))
        left_will_hit = self._hit_something(obs_board, (agent_a_head[0], agent_a_head[1] - 1))
        right_will_hit = self._hit_something(obs_board, (agent_a_head[0], agent_a_head[1] + 1))
        
        if not up_will_hit:
            valid_actions.append(Snake.MOVE_UP)
        if not down_will_hit:
            valid_actions.append(Snake.MOVE_DOWN)
        if not left_will_hit:
            valid_actions.append(Snake.MOVE_LEFT)
        if not right_will_hit:
            valid_actions.append(Snake.MOVE_RIGHT)

        return random.choice(valid_actions) if len(valid_actions) > 0 else Snake.MOVE_UP

    def learn(self, *args, **kwargs):
        return
        
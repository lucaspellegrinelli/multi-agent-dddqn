import numpy as np

class Snake:
    AGENT_A_ID = 0
    AGENT_B_ID = 1

    EMPTY_SQUARE = -1
    AGENT_A_HEAD_ID = 0
    AGENT_B_HEAD_ID = 1
    AGENT_A_BODY_ID = 2
    AGENT_B_BODY_ID = 3
    FOOD_ID = 4

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3

    BOARD_SIZE = 10

    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps
        self.board_size = self.BOARD_SIZE
        self.reset()

    def reset(self):
        self.board = np.ones((self.board_size, self.board_size), dtype=np.int32) * self.EMPTY_SQUARE
        self.board[0, 0] = self.AGENT_A_HEAD_ID
        self.board[self.board_size - 1, self.board_size - 1] = self.AGENT_B_HEAD_ID
        self.step_count = 0
        self.add_food()

        self.agent_foods = { self.AGENT_A_ID: 0, self.AGENT_B_ID: 0 }
        self.agent_alives = { self.AGENT_A_ID: True, self.AGENT_B_ID: True }
        self.agent_positions = {
            self.AGENT_A_ID: [(0, 0)],
            self.AGENT_B_ID: [(self.board_size - 1, self.board_size - 1)]
        }

    def step(self, agent: int, action: int) -> int:
        if not self.agent_alives[agent]:
            return -1

        self.step_count += 1

        # invert action if agent is agent B
        if agent == self.AGENT_B_ID:
            if action == self.MOVE_UP:
                action = self.MOVE_DOWN
            elif action == self.MOVE_DOWN:
                action = self.MOVE_UP

        if action == self.MOVE_UP:
            new_position = (self.agent_positions[agent][-1][0] - 1, self.agent_positions[agent][-1][1])
        elif action == self.MOVE_DOWN:
            new_position = (self.agent_positions[agent][-1][0] + 1, self.agent_positions[agent][-1][1])
        elif action == self.MOVE_LEFT:
            new_position = (self.agent_positions[agent][-1][0], self.agent_positions[agent][-1][1] - 1)
        elif action == self.MOVE_RIGHT:
            new_position = (self.agent_positions[agent][-1][0], self.agent_positions[agent][-1][1] + 1)
        
        # check if new position is valid
        if new_position[0] < 0 or new_position[0] >= self.board_size or new_position[1] < 0 or new_position[1] >= self.board_size:
            self.agent_alives[agent] = False
            return -1

        # check if new position is food
        ate_food = False
        if self.board[new_position] == self.FOOD_ID:
            ate_food = True
            self.agent_foods[agent] += 1

        # check if new position is other agent
        new_pos_content = self.board[new_position[0], new_position[1]]
        is_other_agent_head = new_pos_content == self.AGENT_A_HEAD_ID or new_pos_content == self.AGENT_B_HEAD_ID
        is_other_agent_body = new_pos_content == self.AGENT_A_BODY_ID or new_pos_content == self.AGENT_B_BODY_ID
        if is_other_agent_head or is_other_agent_body:
            self.agent_alives[agent] = False
            return -1

        # update board
        prev_head_pos = self.agent_positions[agent][-1]
        self.agent_positions[agent].append(new_position)
        self.board[new_position] = self.AGENT_A_HEAD_ID if agent == self.AGENT_A_ID else self.AGENT_B_HEAD_ID
        self.board[prev_head_pos] = self.AGENT_A_BODY_ID if agent == self.AGENT_A_ID else self.AGENT_B_BODY_ID

        if ate_food:
            self.add_food()
        else:
            self.board[self.agent_positions[agent][0]] = self.EMPTY_SQUARE
            self.agent_positions[agent].pop(0)
        
        return 1 if ate_food else 0
        
    def add_food(self):
        while True:
            food_position = (np.random.randint(0, self.board_size), np.random.randint(0, self.board_size))
            if self.board[food_position] == self.EMPTY_SQUARE:
                self.board[food_position] = self.FOOD_ID
                break

    def observation(self, agent: int):
        if agent == self.AGENT_A_ID:
            return np.expand_dims(self.board, axis=0).copy()
        elif agent == self.AGENT_B_ID:
            flipped = np.flip(self.board.copy(), axis=0)
            flipped_copy = flipped.copy()

            # flip agent heads
            flipped[flipped_copy == self.AGENT_A_HEAD_ID] = self.AGENT_B_HEAD_ID
            flipped[flipped_copy == self.AGENT_B_HEAD_ID] = self.AGENT_A_HEAD_ID

            # flip agent bodies
            flipped[flipped_copy == self.AGENT_A_BODY_ID] = self.AGENT_B_BODY_ID
            flipped[flipped_copy == self.AGENT_B_BODY_ID] = self.AGENT_A_BODY_ID

            return np.expand_dims(flipped, axis=0)

    def is_game_ended(self):
        return self.step_count > self.n_steps or not self.all_agents_alive()

    def all_agents_alive(self):
        return all([self.agent_alives[agent] for agent in self.agent_alives])

    def get_score(self, agent: int):
        return self.agent_foods[agent]

    def get_winner(self):
        if self.is_game_ended():
            if not self.all_agents_alive():
                return self.AGENT_A_ID if self.agent_alives[self.AGENT_A_ID] else self.AGENT_B_ID
            else:
                return self.AGENT_A_ID if self.agent_foods[self.AGENT_A_ID] > self.agent_foods[self.AGENT_B_ID] else self.AGENT_B_ID
        
        return None

    @staticmethod
    def action_size():
        return 4 # up, down, left, right

    @staticmethod
    def observation_size():
        return (1, Snake.BOARD_SIZE, Snake.BOARD_SIZE)

    def __repr__(self):
        return str(self.board)

    
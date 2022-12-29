import random
import numpy as np

class TicTacToe:
    PLAY_ACTION = 0
    CHECK_ACTION = 1

    CLEAR_PROB = -1
    CLEAR_ID = -1
    AGENT_X_ID = 0
    AGENT_O_ID = 1

    TURN_REWARD = 0
    INVALID_REWARD = -1
    LOSE_REWARD = -1
    DRAW_REWARD = 0
    WIN_REWARD = 1

    def __init__(self, deterministic: bool = False):
        self.deterministic = deterministic
        self.reset()

    @staticmethod
    def action_size():
        return 18

    @staticmethod
    def observation_size():
        return 9 + 9 + 1

    def reset(self):
        self.who_invalidated = []
        self.squares = [self.CLEAR_ID for _ in range(9)]
        self.square_probs = [random.uniform(0.25, 1) for _ in range(9)]
        self.agent_knowledge = {id: [self.CLEAR_PROB for _ in range(9)] for id in [self.AGENT_X_ID, self.AGENT_O_ID]}

        self.agent_reward = {id: 0 for id in [self.AGENT_X_ID, self.AGENT_O_ID]}

    def step(self, agent: int, action: int):
        action_type = self.PLAY_ACTION if action < 9 else self.CHECK_ACTION
        action_pos = action % 9

        if self.is_game_ended():
            if len(self.who_invalidated) > 0:
                if agent in self.who_invalidated:
                    reward = self.INVALID_REWARD
                elif agent not in self.who_invalidated:
                    reward = self.WIN_REWARD
            elif self.is_draw():
                reward = self.DRAW_REWARD
            elif self.has_won(agent):
                reward = self.WIN_REWARD
            else:
                reward = self.LOSE_REWARD

        is_valid = self.play_turn(agent, action_type, action_pos)

        done = self.is_game_ended()

        if not is_valid:
            self.who_invalidated.append(agent)
            reward = self.INVALID_REWARD
        elif done and not self.is_invalidated():
            if self.is_draw():
                reward = self.DRAW_REWARD
            elif self.has_won(agent):
                reward = self.WIN_REWARD
            else:
                reward = self.LOSE_REWARD
        else:
            reward = self.TURN_REWARD

        self.agent_reward[agent] = reward

    def get_view(self, agent: int):
        state = self.get_state(agent)
        reward = self.agent_reward[agent] if agent not in self.who_invalidated else self.INVALID_REWARD
        done = self.is_game_ended()
        return state, reward, done

    def get_state(self, agent: int):
        return np.array(self.squares + self.agent_knowledge[agent] + [agent], dtype=np.float32)

    def play_turn(self, agent: int, action: int, pos: int):
        if pos < 0 or pos >= 9:
            return False

        if agent != self.AGENT_X_ID and agent != self.AGENT_O_ID:
            return False

        if action != self.PLAY_ACTION and action != self.CHECK_ACTION:
            return False

        if action == self.PLAY_ACTION:
            if self.squares[pos] != self.CLEAR_ID:
                return False

            if random.random() < self.square_probs[pos] or self.deterministic:
                self.squares[pos] = agent
            return True
        elif action == self.CHECK_ACTION:
            if self.agent_knowledge[agent][pos] != self.CLEAR_PROB:
                return False

            self.agent_knowledge[agent][pos] = self.square_probs[pos]
            return True

    def has_won(self, agent: int):
        def check_line(a: int, b: int, c: int):
            return self.squares[a] == self.squares[b] == self.squares[c] == agent

        return check_line(0, 1, 2) or check_line(3, 4, 5) or check_line(6, 7, 8) or \
            check_line(0, 3, 6) or check_line(1, 4, 7) or check_line(2, 5, 8) or \
            check_line(0, 4, 8) or check_line(2, 4, 6)

    def is_game_ended(self):
        return all(square in [self.AGENT_X_ID, self.AGENT_O_ID] for square in self.squares) or len(self.who_invalidated) > 0 or self.has_won(self.AGENT_X_ID) or self.has_won(self.AGENT_O_ID)

    def is_draw(self):
        return self.is_game_ended() and not self.has_won(self.AGENT_X_ID) and not self.has_won(self.AGENT_O_ID) and len(self.who_invalidated) == 0

    def is_invalidated(self):
        return len(self.who_invalidated) > 0

    def all_squares_played(self):
        return all(square in [self.AGENT_X_ID, self.AGENT_O_ID] for square in self.squares)

    def __repr__(self):
        out_repr = ""

        def getSymbol(input):
            if input == self.CLEAR_ID:
                return "-"
            elif input == self.AGENT_X_ID:
                return "X"
            elif input == self.AGENT_O_ID:
                return "O"

        board = list(map(getSymbol, self.squares))

        for i in range(3):
            out_repr += "|"
            for j in range(3):
                out_repr += f"-------|"
            out_repr += "\n"

            out_repr += "|"
            for j in range(3):
                idx = j * 3 + i
                out_repr += f"{self.square_probs[idx]:6.3f} |"
            out_repr += "\n"

            out_repr += "|"
            for j in range(3):
                idx = j * 3 + i
                out_repr += f"   {board[idx]}   |"
            out_repr += "\n"

            out_repr += "|"
            for j in range(3):
                p1 = "✓" if self.agent_knowledge[0][j * 3 + i] != -1 else "✗"
                p2 = "✓" if self.agent_knowledge[1][j * 3 + i] != -1 else "✗"
                out_repr += f"  {p1} {p2}  |"
            out_repr += "\n"

        out_repr += "|"
        for j in range(3):
            out_repr += f"-------|"
        out_repr += "\n"

        return out_repr

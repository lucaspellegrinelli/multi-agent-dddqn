import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.base import Agent

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, n_actions: int, lr: float = 1e-4):
        super(DuelingDeepQNetwork, self).__init__()

        # create a convolutional sequential model
        self.model = nn.Sequential(
            # 10 x 10 x 1
            nn.Conv2d(1, 32, kernel_size=3), # 8 x 8 x 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), # 6 x 6 x 32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), # 4 x 4 x 32
            nn.ReLU(),
            nn.Flatten(), # 512
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state: np.ndarray):
        x = self.model(state)
        return self.V(x), self.A(x)


class DDDQNAgent(Agent):
    def __init__(
        self,
        n_actions: int,
        observation_shape: list[int] | tuple[int],
        qnet: DuelingDeepQNetwork,
        target_qnet: DuelingDeepQNetwork,
        batch_size: int = 64,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        memory_size: int = 100000,
        eps_min: float = 0,
        eps_dec: float = 1e-4,
        target_qnet_update_freq: int = 100
    ):
        # Environment parameters
        self.n_actions = n_actions
        self.observation_shape = observation_shape

        # Learning parameters
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        # Network parameters
        self.epoch = 0
        self.qnet = qnet
        self.target_qnet = target_qnet
        self.target_qnet_update_freq = target_qnet_update_freq

        # Memory / Replay Buffer
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((memory_size, *observation_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((memory_size, *observation_shape), dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(memory_size, dtype=np.bool_)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def act(self, observation: np.ndarray):
        assert observation.shape == tuple(self.observation_shape), f"Observation shape {observation.shape} does not match expected shape {self.observation_shape}"

        if np.random.random() > self.epsilon:
            state = torch.tensor(np.expand_dims(observation.copy(), axis=0), dtype=torch.float).to(self.device)
            _, adv = self.qnet.forward(state)
            return torch.argmax(adv).item()

        return np.random.randint(0, self.n_actions)

    def store_memory(self, prev_state: np.ndarray, action: int, reward: float, new_state: np.ndarray, done: bool):
        assert prev_state.shape == tuple(self.observation_shape), f"Observation shape {prev_state.shape} does not match expected shape {self.observation_shape}"
        assert new_state.shape == tuple(self.observation_shape), f"Observation shape {new_state.shape} does not match expected shape {self.observation_shape}"
        assert action in range(self.n_actions), f"Action {action} is not in range of possible actions {self.n_actions}"

        memory_index = self.memory_counter % self.memory_size
        self.state_memory[memory_index] = prev_state
        self.new_state_memory[memory_index] = new_state
        self.action_memory[memory_index] = action
        self.reward_memory[memory_index] = reward
        self.terminal_memory[memory_index] = done
        self.memory_counter = min(self.memory_counter + 1, self.memory_size)

    def sample_memory(self, batch_size: int):
        memory_indexes = np.random.choice(self.memory_counter, batch_size, replace=False)
        prev_states = self.state_memory[memory_indexes]
        actions = self.action_memory[memory_indexes]
        rewards = self.reward_memory[memory_indexes]
        new_states = self.new_state_memory[memory_indexes]
        terminal = self.terminal_memory[memory_indexes]
        return prev_states, actions, rewards, new_states, terminal

    def save(self, qnet_path: str, target_qnet_path: str):
        torch.save(self.qnet.state_dict(), qnet_path)
        torch.save(self.target_qnet.state_dict(), target_qnet_path)

    def load(self, qnet_path: str, target_qnet_path: str):
        self.qnet.load_state_dict(torch.load(qnet_path))
        self.target_qnet.load_state_dict(torch.load(target_qnet_path))

    def learn(self, prev_state: np.ndarray, action: int, reward: float, new_state: np.ndarray, done: int | bool):
        # Store the transition in memory
        self.store_memory(prev_state, action, reward, new_state, done)

        # If we don't have enough samples in memory for a batch, we can't learn yet
        if self.memory_counter < self.batch_size:
            return

        self.qnet.optimizer.zero_grad()

        # Update the target network if it's time
        self.epoch += 1
        if self.epoch % self.target_qnet_update_freq == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        # Sample a batch of transitions from memory
        prev_state, action, reward, new_state, done = self.sample_memory(self.batch_size)

        # Convert the batch to tensors
        prev_states = torch.tensor(prev_state).to(self.device)
        rewards = torch.tensor(reward).to(self.device)
        dones = torch.tensor(done).to(self.device)
        actions = torch.tensor(action).to(self.device)
        new_states = torch.tensor(new_state).to(self.device)

        # Calculate the target Q values for the previous states using the Q network
        qnet_V, qnet_A = self.qnet.forward(prev_states)

        # Calculate the target Q values for the new states using the target Q network
        target_V, target_A = self.target_qnet.forward(new_states)

        # Predict the Q values for the new states using the Q network
        pred_qnet_V, pred_qnet_A = self.qnet.forward(new_states)

        # Calculate the Q values for the previous states using the Q network
        indices = np.arange(self.batch_size)
        q_pred = torch.add(qnet_V, (qnet_A - qnet_A.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(target_V, (target_A - target_A.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0
        q_eval = torch.add(pred_qnet_V, (pred_qnet_A - pred_qnet_A.mean(dim=1, keepdim=True)))

        # Calculate the target Q values
        max_actions = torch.argmax(q_eval, dim=1)
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # Calculate the loss between the predicted and target Q values
        loss = self.qnet.loss(q_target, q_pred).to(self.device)

        # Backpropagate the loss
        loss.backward()
        self.qnet.optimizer.step()

        # Update the epsilon value
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

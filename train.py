import os
import torch
import random
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents import RandomAgent, DDDQNAgent, Agent
from agents.dddqnagent import DuelingDeepQNetwork
from envirorment import TicTacToe

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true", help="Load models from disk")
parser.add_argument("--save", action="store_true", help="Save models to disk")
parser.add_argument("--num_games", type=int, default=50000, help="Number of games to play")
args = parser.parse_args()

def build_agent(cold_start: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_actions = TicTacToe.action_size()
    obs_size = TicTacToe.observation_size()
    return DDDQNAgent(
        n_actions=n_actions,
        observation_shape=[obs_size],
        qnet=DuelingDeepQNetwork(n_actions=n_actions, input_shape=[obs_size], lr=5e-6).to(device),
        target_qnet=DuelingDeepQNetwork(n_actions=n_actions, input_shape=[obs_size], lr=5e-6).to(device),
        batch_size=64,
        target_qnet_update_freq=2500,
        epsilon=1.0 if cold_start else 0.0,
        eps_dec=1e-4,
        eps_min=0,
    )

def play_game(agent_x: Agent, agent_o: Agent):
    env = TicTacToe()

    is_x_turn = True
    while not env.is_game_ended():
        agent_id = TicTacToe.AGENT_X_ID if is_x_turn else TicTacToe.AGENT_O_ID
        agent = agent_x if is_x_turn else agent_o

        obs, _, _ = env.get_view(agent_id)
        action = agent.act(obs)
        if action is not None:
            env.step(agent_id, action)

        is_x_turn = not is_x_turn

    return {
        "draw": env.is_draw(),
        "x_won": env.has_won(TicTacToe.AGENT_X_ID),
        "o_won": env.has_won(TicTacToe.AGENT_O_ID),
        "invalid": env.is_invalidated()
    }

def evaluate(agent_a: Agent, agent_b: Agent, n_games: int, shuffle: bool = True):
    results = { "draw": 0, "a_won": 0, "b_won": 0, "invalid": 0 }

    for _ in range(n_games):
        a_as_x = not shuffle or np.random.uniform(0, 1) < 0.5
        result = play_game(
            agent_x=agent_a if a_as_x else agent_b,
            agent_o=agent_b if a_as_x else agent_a
        )
        
        results["draw"] += result["draw"]
        results["a_won"] += result["x_won"] if a_as_x else result["o_won"]
        results["b_won"] += result["o_won"] if a_as_x else result["x_won"]
        results["invalid"] += result["invalid"]

    return { key: value / n_games for key, value in results.items() }

if __name__ == "__main__":
    writer = SummaryWriter()

    agent_a = build_agent(not args.load)
    agent_b = build_agent(not args.load)

    if args.load:
        agent_a.load("models/agent_a/qnet.pth", "models/agent_a/target_qnet.pth")
        agent_b.load("models/agent_b/qnet.pth", "models/agent_b/target_qnet.pth")

    for game_i in range(args.num_games):
        env = TicTacToe()

        # Randomly choose players for this game
        agent_x = random.choice([agent_a, agent_b, RandomAgent(18)])
        agent_o = random.choice([agent_a, agent_b, RandomAgent(18)])

        while True:
            if not env.is_game_ended():
                pre_observation_x, _, _ = env.get_view(TicTacToe.AGENT_X_ID)
                action_x = agent_x.act(pre_observation_x)
                env.step(TicTacToe.AGENT_X_ID, action_x)

            if not env.is_game_ended():
                pre_observation_o, _, _ = env.get_view(TicTacToe.AGENT_O_ID)
                action_o = agent_o.act(pre_observation_o)
                env.step(TicTacToe.AGENT_O_ID, action_o)

            observation_x, reward_x, done_x, = env.get_view(TicTacToe.AGENT_X_ID)
            observation_o, reward_o, done_o, = env.get_view(TicTacToe.AGENT_O_ID)

            agent_x.learn(pre_observation_x, action_x, reward_x, observation_x, int(done_x))
            agent_o.learn(pre_observation_o, action_o, reward_o, observation_o, int(done_o))

            if done_x or done_o:
                break

        if game_i % 1000 == 0:
            if game_i != 0 and args.save:
                os.makedirs("models/agent_a", exist_ok=True)
                os.makedirs("models/agent_b", exist_ok=True)
                agent_a.save("models/agent_a/qnet.pth", "models/agent_a/target_qnet.pth")
                agent_b.save("models/agent_b/qnet.pth", "models/agent_b/target_qnet.pth")
            
            a_vs_b = evaluate(agent_a, agent_b, n_games=1000)
            a_vs_rnd = evaluate(agent_a, RandomAgent(18), n_games=1000)
            b_vs_rnd = evaluate(agent_b, RandomAgent(18), n_games=1000)

            writer.add_scalar("a_vs_b_draw", a_vs_b["draw"], game_i)
            writer.add_scalar("a_vs_b_a_won", a_vs_b["a_won"], game_i)
            writer.add_scalar("a_vs_b_b_won", a_vs_b["b_won"], game_i)
            writer.add_scalar("a_vs_b_invalid", a_vs_b["invalid"], game_i)
            writer.add_scalar("a_vs_rnd_draw", a_vs_rnd["draw"], game_i)
            writer.add_scalar("a_vs_rnd_a_won", a_vs_rnd["a_won"], game_i)
            writer.add_scalar("a_vs_rnd_rnd_won", a_vs_rnd["b_won"], game_i)
            writer.add_scalar("a_vs_rnd_invalid", a_vs_rnd["invalid"], game_i)
            writer.add_scalar("b_vs_rnd_draw", b_vs_rnd["draw"], game_i)
            writer.add_scalar("b_vs_rnd_b_won", b_vs_rnd["a_won"], game_i)
            writer.add_scalar("b_vs_rnd_rnd_won", b_vs_rnd["b_won"], game_i)
            writer.add_scalar("b_vs_rnd_invalid", b_vs_rnd["invalid"], game_i)
            writer.add_scalar("epsilon_a", agent_a.epsilon, game_i)
            writer.add_scalar("epsilon_b", agent_b.epsilon, game_i)

            print(f"Episode {game_i:5d}")
            print(f" > Epsilon A   {agent_a.epsilon:5.2f}  | Epsilon B   {agent_b.epsilon:5.2f}")
            print(f" > Agent A vs Agent B | Draw {a_vs_b['draw']:5.2f} | A Won {a_vs_b['a_won']:5.2f} | B Won {a_vs_b['b_won']:5.2f} | Invalid {a_vs_b['invalid']:5.2f}")
            print(f" > Agent A vs Random  | Draw {a_vs_rnd['draw']:5.2f} | A Won {a_vs_rnd['a_won']:5.2f} | B Won {a_vs_rnd['b_won']:5.2f} | Invalid {a_vs_rnd['invalid']:5.2f}")
            print(f" > Agent B vs Random  | Draw {b_vs_rnd['draw']:5.2f} | A Won {b_vs_rnd['a_won']:5.2f} | B Won {b_vs_rnd['b_won']:5.2f} | Invalid {b_vs_rnd['invalid']:5.2f}")
            print("")

    writer.flush()
    writer.close()
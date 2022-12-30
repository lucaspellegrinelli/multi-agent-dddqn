import os
import torch
import random
import argparse
import numpy as np
import concurrent.futures
from torch.utils.tensorboard import SummaryWriter

from agents import SnakeRandomAgent, DDDQNAgent, Agent
from agents.dddqnagent import DuelingDeepQNetwork
from envirorment import Snake

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true", help="Load models from disk")
parser.add_argument("--save", action="store_true", help="Save models to disk")
parser.add_argument("--num_games", type=int, default=1000000, help="Number of games to play")
args = parser.parse_args()

def build_agent(cold_start: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_actions = Snake.action_size()
    obs_size = Snake.observation_size()

    return DDDQNAgent(
        n_actions=n_actions,
        observation_shape=obs_size,
        qnet=DuelingDeepQNetwork(n_actions=n_actions, lr=1e-4).to(device),
        target_qnet=DuelingDeepQNetwork(n_actions=n_actions, lr=1e-4).to(device),
        batch_size=64,
        target_qnet_update_freq=5000,
        epsilon=1.0 if cold_start else 0.0,
        eps_dec=1e-4,
        eps_min=0,
    )

def play_game(agent_a: Agent, agent_b: Agent):
    env = Snake(n_steps=250)

    info = { "a": [], "b": [] }

    while True:
        pre_obs_a = env.observation(env.AGENT_A_ID)
        action_a = agent_a.act(pre_obs_a)
        reward_a = env.step(env.AGENT_A_ID, action_a)

        pre_obs_b = env.observation(env.AGENT_B_ID)
        action_b = agent_b.act(pre_obs_b)
        reward_b = env.step(env.AGENT_B_ID, action_b)

        obs_a = env.observation(env.AGENT_A_ID)
        obs_b = env.observation(env.AGENT_B_ID)

        done = env.is_game_ended()

        info["a"].append({ "pre_obs": pre_obs_a, "action": action_a, "reward": reward_a, "obs": obs_a, "done": done })
        info["b"].append({ "pre_obs": pre_obs_b, "action": action_b, "reward": reward_b, "obs": obs_b, "done": done })

        if done:
            break

    return {
        "a_won": env.get_winner() == env.AGENT_A_ID,
        "b_won": env.get_winner() == env.AGENT_B_ID,
        "steps": env.step_count,
        "a_score": env.get_score(env.AGENT_A_ID),
        "b_score": env.get_score(env.AGENT_B_ID),
        "info": info,
    }

def evaluate(agent_a: Agent, agent_b: Agent, n_games: int, shuffle: bool = True):
    results = { "a_won": 0, "b_won": 0, "a_score": 0, "b_score": 0, "steps": 0 }

    for _ in range(n_games):
        a_first = not shuffle or np.random.uniform(0, 1) < 0.5
        result = play_game(
            agent_a=agent_a if a_first else agent_b,
            agent_b=agent_b if a_first else agent_a
        )
        
        results["a_won"] += result["a_won"] if a_first else result["b_won"]
        results["b_won"] += result["b_won"] if a_first else result["a_won"]
        results["a_score"] += result["a_score"]
        results["b_score"] += result["b_score"]
        results["steps"] += result["steps"]

    return { key: value / n_games for key, value in results.items() }

if __name__ == "__main__":
    N_PROCESSES = 5
    writer = SummaryWriter()

    agent_0 = build_agent(not args.load)
    agent_1 = build_agent(not args.load)

    if args.load:
        agent_0.load("models/agent_a/qnet.pth", "models/agent_a/target_qnet.pth")
        agent_1.load("models/agent_b/qnet.pth", "models/agent_b/target_qnet.pth")

    for game_i in range(0, args.num_games, N_PROCESSES):
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
            # Randomly choose players for this game
            player_a = random.choice([agent_0, agent_1, SnakeRandomAgent()])
            player_b = random.choice([agent_0, agent_1, SnakeRandomAgent()])

            # Play games
            game_infos = executor.map(play_game, [player_a] * N_PROCESSES, [player_b] * N_PROCESSES)

            # Update agents
            for game_info in game_infos:
                for info_a, info_b in zip(game_info["info"]["a"], game_info["info"]["b"]):
                    player_a.learn(info_a["pre_obs"], info_a["action"], info_a["reward"], info_a["obs"], info_a["done"])
                    player_b.learn(info_b["pre_obs"], info_b["action"], info_b["reward"], info_b["obs"], info_b["done"])

        if game_i % 1000 == 0:
            if game_i != 0 and args.save:
                os.makedirs("models/agent_a", exist_ok=True)
                os.makedirs("models/agent_b", exist_ok=True)
                agent_0.save("models/agent_a/qnet.pth", "models/agent_a/target_qnet.pth")
                agent_1.save("models/agent_b/qnet.pth", "models/agent_b/target_qnet.pth")
            
            a_vs_b = evaluate(agent_0, agent_1, n_games=100)
            a_vs_rnd = evaluate(agent_0, SnakeRandomAgent(), n_games=100)
            b_vs_rnd = evaluate(agent_1, SnakeRandomAgent(), n_games=100)
            rnd_vs_rnd = evaluate(SnakeRandomAgent(), SnakeRandomAgent(), n_games=100)

            writer.add_scalar("a_vs_b_a_won", a_vs_b["a_won"], game_i)
            writer.add_scalar("a_vs_b_b_won", a_vs_b["b_won"], game_i)
            writer.add_scalar("a_vs_b_a_score", a_vs_b["a_score"], game_i)
            writer.add_scalar("a_vs_b_b_score", a_vs_b["b_score"], game_i)
            writer.add_scalar("a_vs_b_steps", a_vs_b["steps"], game_i)

            writer.add_scalar("a_vs_rnd_a_won", a_vs_rnd["a_won"], game_i)
            writer.add_scalar("a_vs_rnd_rnd_won", a_vs_rnd["b_won"], game_i)
            writer.add_scalar("a_vs_rnd_a_score", a_vs_rnd["a_score"], game_i)
            writer.add_scalar("a_vs_rnd_rnd_score", a_vs_rnd["b_score"], game_i)
            writer.add_scalar("a_vs_rnd_steps", a_vs_rnd["steps"], game_i)

            writer.add_scalar("b_vs_rnd_b_won", b_vs_rnd["a_won"], game_i)
            writer.add_scalar("b_vs_rnd_rnd_won", b_vs_rnd["b_won"], game_i)
            writer.add_scalar("b_vs_rnd_b_score", b_vs_rnd["a_score"], game_i)
            writer.add_scalar("b_vs_rnd_rnd_score", b_vs_rnd["b_score"], game_i)
            writer.add_scalar("b_vs_rnd_steps", b_vs_rnd["steps"], game_i)

            writer.add_scalar("epsilon_a", agent_0.epsilon, game_i)
            writer.add_scalar("epsilon_b", agent_1.epsilon, game_i)

            print(f"Episode {game_i:5d}")
            print(f" > Epsilon A    {agent_0.epsilon:5.2f} | Epsilon B   {agent_1.epsilon:5.2f}")
            print(f" > Agent A vs Agent B | A Won {a_vs_b['a_won']:5.2f} | B Won {a_vs_b['b_won']:5.2f} | A Score {a_vs_b['a_score']:5.2f} | B Score {a_vs_b['b_score']:5.2f} | Steps {a_vs_b['steps']:6.2f}")
            print(f" > Agent A vs Random  | A Won {a_vs_rnd['a_won']:5.2f} | B Won {a_vs_rnd['b_won']:5.2f} | A Score {a_vs_rnd['a_score']:5.2f} | B Score {a_vs_rnd['b_score']:5.2f} | Steps {a_vs_rnd['steps']:6.2f}")
            print(f" > Agent B vs Random  | A Won {b_vs_rnd['a_won']:5.2f} | B Won {b_vs_rnd['b_won']:5.2f} | A Score {b_vs_rnd['a_score']:5.2f} | B Score {b_vs_rnd['b_score']:5.2f} | Steps {b_vs_rnd['steps']:6.2f}")
            print(f" > Random vs Random   | A Won {rnd_vs_rnd['a_won']:5.2f} | B Won {rnd_vs_rnd['b_won']:5.2f} | A Score {rnd_vs_rnd['a_score']:5.2f} | B Score {rnd_vs_rnd['b_score']:5.2f} | Steps {rnd_vs_rnd['steps']:6.2f}")
            print("")

    writer.flush()
    writer.close()
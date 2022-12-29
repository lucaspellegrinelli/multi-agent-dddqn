import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")

parser = argparse.ArgumentParser(description="Train visualization")
parser.add_argument("--path", type=str, help="Path to tensorboard logs")
parser.add_argument("--out", type=str, default="plots", help="Path to output directory")
args = parser.parse_args()

def plot_head_to_head(summary: EventAccumulator, title: str, name_a: str, name_b: str):
    _, step, draw = zip(*summary.Scalars(f"{name_a}_vs_{name_b}_draw"))
    _, step, invalid = zip(*summary.Scalars(f"{name_a}_vs_{name_b}_invalid"))
    _, step, win_a = zip(*summary.Scalars(f"{name_a}_vs_{name_b}_{name_a}_won"))
    _, step, win_b = zip(*summary.Scalars(f"{name_a}_vs_{name_b}_{name_b}_won"))
    
    os.makedirs(args.out, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(step, invalid, label="Invalid", color="#95a5a6", linewidth=3)
    plt.plot(step, draw, label="Draw", color="#f1c40f", linewidth=3)
    plt.plot(step, win_b, label=f"{name_b} won", color="#e74c3c", linewidth=3)
    plt.plot(step, win_a, label=f"{name_a} won", color="#1abc9c", linewidth=3)

    plt.xlabel("Number of games")
    plt.ylabel("Percentage")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{name_a}_vs_{name_b}.png"))
    plt.clf()

if __name__ == "__main__":
    summary = EventAccumulator(args.path)
    summary.Reload()
    
    plot_head_to_head(summary, "Agent A vs Agent B", "a", "b")
    plot_head_to_head(summary, "Agent A vs Random Agent", "a", "rnd")
    plot_head_to_head(summary, "Agent B vs Random Agent", "b", "rnd")
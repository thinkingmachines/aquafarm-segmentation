import argparse

import wandb
from setup_wandb_sweep import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep_id", help="input the Wandb Sweep ID to run the Wandb agent"
    )
    parser.add_argument(
        "--count", type=int, help="Set the maximum number of runs the agent will try"
    )
    args = parser.parse_args()
    sweep_id = args.sweep_id
    count = args.count
    wandb.agent(sweep_id, function=train_model, count=count)

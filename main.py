import os
import wandb
from time import sleep

import torch.nn as nn
import torch.optim as optim

from rich import print as rich_print

from utils import seed_everything
from data import create_dataloaders
from choices import choice_1, choice_2, choice_3


if __name__ == "__main__":

    rich_print(
        "[yellow] ___       __   _______   ___  ________  ___  ___  _________  ________           ________             ________  ___  ________  ________  _______   ________      "
    )
    rich_print(
        "[yellow]|\  \     |\  \|\  ___ \ |\  \|\   ____\|\  \|\  \|\___   ___\\   ____\         |\   __  \           |\   __  \|\  \|\   __  \|\   ____\|\  ___ \ |\   ____\     "
    )
    rich_print(
        "[yellow]\ \  \    \ \  \ \   __/|\ \  \ \  \___|\ \  \\\  \|___ \  \_\ \  \___|_        \ \  \|\  \  /\      \ \  \|\ /\ \  \ \  \|\  \ \  \___|\ \   __/|\ \  \___|_    "
    )
    rich_print(
        "[yellow] \ \  \  __\ \  \ \  \_|/_\ \  \ \  \  __\ \   __  \   \ \  \ \ \_____  \        \ \__     \/  \      \ \   __  \ \  \ \   __  \ \_____  \ \  \_|/_\ \_____  \   "
    )
    rich_print(
        "[yellow]  \ \  \|\__\_\  \ \  \_|\ \ \  \ \  \|\  \ \  \ \  \   \ \  \ \|____|\  \        \|_/  __     /|      \ \  \|\  \ \  \ \  \ \  \|____|\  \ \  \_|\ \|____|\  \  "
    )
    rich_print(
        "[yellow]   \ \____________\ \_______\ \__\ \_______\ \__\ \__\   \ \__\  ____\_\  \         /  /_|\   / /       \ \_______\ \__\ \__\ \__\____\_\  \ \_______\____\_\  \ "
    )
    rich_print(
        "[yellow]    \|____________|\|_______|\|__|\|_______|\|__|\|__|    \|__| |\_________\       /_______   \/         \|_______|\|__|\|__|\|__|\_________\|_______|\_________\ "
    )
    rich_print(
        "[yellow]                                                                \|_________|       |_______|\__\                                 \|_________|        \|_________|"
    )
    rich_print(
        "[yellow]                                                                                          \|__|                                                                 "
    )

    sleep(3)

    rich_print("[yellow]Please Select the following Choices:")
    rich_print("[cyan]\t1. [yellow]Train from Scratch")
    rich_print("[cyan]\t2. [yellow]Train from Pre-trained Checkpoint")
    rich_print("[cyan]\t3. [yellow]Do Both")
    choice = int(input())
    print("\n")

    rich_print("[yellow]Please Enter the Number of Training Epochs:")
    epochs = int(input())
    print("\n")

    os.environ["WANDB_CONSOLE"] = "wrap"
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project="common-ml-errors", job_type="pytorch_save_load", anonymous="must"
    )
    run._label("repl_pytorch_save_load")
    config = wandb.config
    config.seed = 42
    config.n_samples = 1000
    config.noise = 0.02
    config.batch_size = 256
    config.hidden_units = 8
    config.learning_rate = 1e-3
    config.epochs = epochs

    seed_everything(config.seed)

    train_dataloader, val_dataloader = create_dataloaders(config)
    train_size, val_size = len(train_dataloader.dataset), len(val_dataloader.dataset)

    model = nn.Sequential(
        nn.Linear(2, 30), nn.ReLU(), nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 1)
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    if choice == 1:
        choice_1(
            run,
            config,
            train_dataloader,
            val_dataloader,
            model,
            loss_function,
            optimizer,
        )
    elif choice == 2:
        choice_2(
            run,
            config,
            train_dataloader,
            val_dataloader,
            model,
            loss_function,
            optimizer,
        )
    elif choice == 3:
        choice_3(
            run,
            config,
            train_dataloader,
            val_dataloader,
            model,
            loss_function,
            optimizer,
        )

    wandb.finish()

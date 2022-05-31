import os
import wandb
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim

from rich.live import Live
from rich.table import Column
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
)

from styles import interface
from data import create_dataloaders
from train import train_and_validate
from utils import seed_everything, shorten_url, cleanup


if __name__ == "__main__":

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
    config.epochs = 10

    seed_everything(config.seed)

    train_dataloader, val_dataloader = create_dataloaders(config)
    train_size, val_size = len(train_dataloader.dataset), len(val_dataloader.dataset)

    model = nn.Sequential(
        nn.Linear(2, 30), nn.ReLU(), nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 1)
    )

    bar_column = BarColumn(bar_width=None, table_column=Column(ratio=1))

    epoch_progress_1 = Progress(
        "[yellow]{task.description}",
        SpinnerColumn(),
        bar_column,
        TimeElapsedColumn(),
        expand=True,
    )
    epoch_progress_2 = Progress(
        "[yellow]{task.description}",
        SpinnerColumn(),
        bar_column,
        TimeElapsedColumn(),
        expand=True,
    )
    epoch_task_1 = epoch_progress_1.add_task(
        "Training", total=config.epochs // 2, start=False
    )
    epoch_task_2 = epoch_progress_2.add_task(
        "Resuming Training", total=config.epochs // 2, start=False
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    layout = interface(epoch_progress_1, f"Training for {config.epochs // 2} epochs")

    with Live(layout, refresh_per_second=20) as live:

        run_url = shorten_url(run.get_url())

        live.update(
            interface(
                epoch_progress_1,
                f"Training for {config.epochs // 2} epochs",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        (
            model,
            loss_history,
            accuracy_history,
            val_loss_history,
            val_accuracy_history,
            epoch_progress_1,
            epoch_task_1,
        ) = train_and_validate(
            model,
            config.batch_size,
            train_dataloader,
            val_dataloader,
            loss_function,
            optimizer,
            config.epochs // 2,
            epoch_progress_1,
            epoch_task_1,
        )

        live.update(
            interface(
                epoch_progress_1,
                "Fetching artifact \[wandb/common-ml-errors/checkpoints:latest]",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        sleep(3)
        artifact = run.use_artifact(
            "wandb/common-ml-errors/checkpoints:latest", type="model"
        )
        artifact_dir = artifact.download()

        live.update(
            interface(
                epoch_progress_1,
                "Loading Checkpoint from Artifact \[checkpoint.pth]",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        sleep(3)
        model_file = os.path.join(artifact_dir, "checkpoint.pth")
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        live.update(
            interface(
                epoch_progress_2,
                f"Resuming Training for {config.epochs // 2} more epochs",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        (
            model,
            loss_history,
            accuracy_history,
            val_loss_history,
            val_accuracy_history,
            epoch_progress_2,
            epoch_task_2,
        ) = train_and_validate(
            model,
            config.batch_size,
            train_dataloader,
            val_dataloader,
            loss_function,
            optimizer,
            config.epochs // 2,
            epoch_progress_2,
            epoch_task_2,
        )

        live.update(
            interface(
                epoch_progress_2,
                "Cleanup in Progress",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        sleep(3)
        cleanup()

        live.update(
            interface(
                epoch_progress_2,
                "Training Ended",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

    wandb.finish()

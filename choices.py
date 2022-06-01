import os
import torch
from time import sleep

from rich.live import Live
from rich.table import Column
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
)

from train import train_and_validate
from utils import shorten_url, cleanup
from styles import interface, final_interface


def choice_1(
    run, config, train_dataloader, val_dataloader, model, loss_function, optimizer
):
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
        "Training", total=config.epochs, start=False
    )
    epoch_task_2 = epoch_progress_2.add_task(
        "Resuming Training", total=config.epochs, start=False
    )

    layout = interface(epoch_progress_1, f"Training for {config.epochs} epochs")

    with Live(layout, refresh_per_second=20) as live:

        run_url = shorten_url(run.get_url())

        live.update(
            interface(
                epoch_progress_1,
                f"Training for {config.epochs} epochs",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        (model, _, _, _, _, epoch_progress_1, epoch_task_1,) = train_and_validate(
            model,
            config.batch_size,
            train_dataloader,
            val_dataloader,
            loss_function,
            optimizer,
            config.epochs,
            epoch_progress_1,
            epoch_task_1,
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
            final_interface(
                epoch_progress_2,
                "Training Ended",
                f"Track your experiment @ {run_url} :bee:",
            )
        )


def choice_2(
    run, config, train_dataloader, val_dataloader, model, loss_function, optimizer
):
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
        "Training", total=config.epochs, start=False
    )
    epoch_task_2 = epoch_progress_2.add_task(
        "Resuming Training", total=config.epochs, start=False
    )

    layout = interface(
        epoch_progress_1,
        "Fetching artifact \[wandb/common-ml-errors/checkpoints:latest]",
    )

    with Live(layout, refresh_per_second=20) as live:

        run_url = shorten_url(run.get_url())

        live.update(
            interface(
                epoch_progress_1,
                "Fetching artifact \[wandb/common-ml-errors/checkpoints:latest]",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        artifact = run.use_artifact(
            "wandb/common-ml-errors/checkpoints:latest", type="model"
        )
        artifact_dir = artifact.download()

        live.update(
            interface(
                epoch_progress_1,
                "Fetching artifact \[wandb/common-ml-errors/checkpoints:latest]",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        sleep(3)
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
                epoch_progress_1,
                f"Training for {config.epochs} more epochs",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        (model, _, _, _, _, epoch_progress_1, epoch_task_1,) = train_and_validate(
            model,
            config.batch_size,
            train_dataloader,
            val_dataloader,
            loss_function,
            optimizer,
            config.epochs,
            epoch_progress_1,
            epoch_task_1,
        )

        live.update(
            interface(
                epoch_progress_1,
                "Cleanup in Progress",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

        sleep(3)
        cleanup()

        live.update(
            final_interface(
                epoch_progress_2,
                "Training Ended",
                f"Track your experiment @ {run_url} :bee:",
            )
        )


def choice_3(
    run, config, train_dataloader, val_dataloader, model, loss_function, optimizer
):
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

        (model, _, _, _, _, epoch_progress_1, epoch_task_1,) = train_and_validate(
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

        (model, _, _, _, _, epoch_progress_2, epoch_task_2,) = train_and_validate(
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
            final_interface(
                epoch_progress_2,
                "Training Ended",
                f"Track your experiment @ {run_url} :bee:",
            )
        )

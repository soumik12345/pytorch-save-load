from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout

from utils import shorten_url


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[b] :bee: Welcome to Weights and Biases :point_right: wandb.ai[/b]",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid, style="black on yellow")


def get_epoch_table(epoch_progress, message: str):
    epoch_table = Table.grid(expand=True)
    epoch_table.add_column(justify="left", ratio=1)
    epoch_table.add_column(justify="right")
    epoch_table.add_row(
        Panel.fit(
            epoch_progress, title=f"[b]{message}", border_style="yellow", padding=(2, 2)
        )
    )

    return epoch_table


def get_footer_table(url=None):
    footer_table = Table.grid(expand=True)
    footer_table.add_column(ratio=1)
    footer_table.add_row(f":point_right: [b]{url}")
    return Panel(footer_table, style="black on yellow")


def interface(
    epoch_progress,
    train_message: str = "Epoch Progression",
    url="Let W&B initialize a run!",
):
    short_url = shorten_url(
        "http://www.wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE"
    )
    report_url = f"For more details Check out our report @ {short_url} :bee:"
    footer_table = get_footer_table(url)
    report_footer_table = get_footer_table(report_url)

    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="epoch-level", size=7),
        Layout(name="footer", size=3),
        Layout(name="report-footer", size=3),
    )

    layout["header"].update(Header())
    layout["epoch-level"].update(get_epoch_table(epoch_progress, train_message))
    layout["footer"].update(footer_table)
    layout["report-footer"].update(report_footer_table)

    return layout


def final_interface(
    epoch_progress,
    train_message: str = "Epoch Progression",
    url="Let W&B initialize a run!",
):
    short_url = shorten_url(
        "http://www.wandb.ai/wandb/common-ml-errors/reports/How-to-Save-and-Load-Models-in-PyTorch--VmlldzozMjg0MTE"
    )
    report_url = f"For more details Check out our report @ {short_url} :bee:"
    footer_table = get_footer_table(url)
    report_footer_table = get_footer_table(report_url)

    goodbye_table = Table.grid(expand=True)
    goodbye_table.add_column(ratio=1)
    goodbye_table.add_row(
        "[b] :point_right: To experience Real-time Experiment Tracking for your own project, log on to wandb.ai/site[/b]"
    )
    goodbye_table = Panel(goodbye_table, style="black on yellow")

    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="epoch-level", size=7),
        Layout(name="footer", size=3),
        Layout(name="report-footer", size=3),
        Layout(name="goodbye-footer", size=3),
    )

    layout["header"].update(Header())
    layout["epoch-level"].update(get_epoch_table(epoch_progress, train_message))
    layout["footer"].update(footer_table)
    layout["report-footer"].update(report_footer_table)
    layout["goodbye-footer"].update(goodbye_table)

    return layout

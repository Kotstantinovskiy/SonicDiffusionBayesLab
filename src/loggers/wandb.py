import logging
import os
from collections import defaultdict

import pandas as pd

import wandb


class WandbLogger:
    def __init__(self, project_name, run_name, run_id=None, config: dict | None = None):
        wandb.login(key=os.environ["WANDB_KEY"].strip())

        self.run = wandb.init(
            id=run_id if run_id else wandb.util.generate_id(),
            project=project_name,
            name=run_name,
            config=config if config else {},
            resume="allow",
        )

    def log_values(self, values: dict, step: int):
        self.run.log(values, step=step)

    def log_images(self, images: dict[str, float]):
        wandb_images = {
            name: [wandb.Image(img) for img in img_list]
            for name, img_list in images.items()
        }
        self.run.log(wandb_images)

    def log_tables(self, tables: dict[str, pd.DataFrame]):
        wandb_tables = {
            name: wandb.Table(dataframe=table) for name, table in tables.items()
        }
        self.run.log(wandb_tables)


class Logger:
    def __init__(
        self,
        config: dict,
        wandb_enable=True,
        project_name: str | None = None,
        run_name: str | None = None,
        run_id: int | None = None,
    ):
        if wandb_enable:
            if project_name is None or run_name is None:
                raise ValueError()

            self.wandb_logger = WandbLogger(
                project_name=project_name,
                run_name=run_name,
                run_id=run_id,
                config=config,
            )

        self.logger = logging.getLogger()
        self.wandb_enable = wandb_enable
        self.losses_history = defaultdict(list)
        self.metrics_history = defaultdict(list)

    def log_metrics(self, metrics: dict, step: int):
        self.wandb_logger.log_values(
            {f"Metrics/{name}": val for name, val in metrics.items()}, step
        )

    def log_metrics_into_table(self, metrics: dict, name_table: str):
        metrics_table = pd.DataFrame.from_dict(metrics, orient="columns")
        self.wandb_logger.log_tables({name_table: metrics_table})

    def log_batch_of_images(
        self,
        images: list,
        name_images: str,
    ):
        self.wandb_logger.log_images(
            {name_images: [image for image in images]},
        )

    """
    def update_losses(self, losses_dict, epoch, step):
        self.losses_memory["Epoch"].append(epoch)
        self.losses_memory["Step"].append(step)
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
    """

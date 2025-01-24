import logging
import os
from collections import defaultdict

import pandas as pd
import wandb


class WandbLogger:
    def __init__(self, project_name, run_name, run_id=None, config: dict | None = None):
        wandb.login(key=os.environ["WANDB_KEY"].strip())

        self.wandb_args = {
            "id": run_id if run_id else wandb.util.generate_id(),
            "project": project_name,
            "name": run_name,
            "config": config if config else {},
            "resume": "allow",
        }

        self.run = wandb.init(**self.wandb_args)

    def log_values(self, values: dict, step: int):
        self.run.log(values, step=step)

    def log_images(self, images: dict[str, float], step: int):
        wandb_images = {
            name: [wandb.Image(img) for img in img_list]
            for name, img_list in images.items()
        }
        self.run.log(wandb_images, step=step)

    def log_tables(self, tables: dict[str, pd.DataFrame], step: int):
        wandb_tables = {
            name: wandb.Table(dataframe=table) for name, table in tables.items()
        }
        self.run.log(wandb_tables, step=step)


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
        self.losses_memory = defaultdict(list)
        self.val_metrics_memory = defaultdict(list)

    def log_metrics(self, val_metrics: dict, step: int):
        self.wandb_logger.log_values(
            {f"metrics/{name}": val for name, val in val_metrics.items()}, step
        )

    def log_metrics_into_table(
        self, val_metrics_table: dict[str, pd.DataFrame], step: int
    ):
        self.wandb_logger.log_tables(
            {f"metrics/{name}": table for name, table in val_metrics_table.items()},
            step,
        )

    def log_batch_of_images(
        self,
        images: list,
        step: int,
    ):
        if not self.wandb_enable:
            self.logger.warning("wandb hasn't been enabled")
            return

        self.wandb_logger.log_images(
            {"Generated images": [image for image in images]},
            step=step,
        )

    """
    def update_losses(self, losses_dict, epoch, step):
        self.losses_memory["Epoch"].append(epoch)
        self.losses_memory["Step"].append(step)
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
    """

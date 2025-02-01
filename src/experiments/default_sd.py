from collections import defaultdict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry, models_registry


@methods_registry.add_to_registry("default")
class DefaultStableDiffusion(BaseMethod):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # setup model
        self.setup_model()

        # setup datasets
        self.setup_dataset()

        # metrics
        self.setup_metrics()

        # loggers
        self.setup_loggers()

        self.num_inference_steps = config.experiment_params.num_inference_steps

    def run_experiment(self):
        batch_size = self.config.inference.get("batch_size", 1)

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        self.metric_dict = defaultdict(list)
        for idx_step, steps in enumerate(self.num_inference_steps):
            self.model.to(self.device)
            gen_images = self.generate(test_dataloader, steps, batch_size)
            self.model.to("cpu")

            gen_dataloader = DataLoader(
                gen_images,
                batch_size=batch_size,
                shuffle=False,
            )

            # update metrics
            self.validate(
                test_dataloader,
                gen_dataloader,
                name_images=f"Inference steps: {steps}",
            )

            self._update_metric_dict(steps)

            self.logger.log_metrics_into_table(
                metrics=self.metric_dict,
                name_table="Default stable diffusion",
            )

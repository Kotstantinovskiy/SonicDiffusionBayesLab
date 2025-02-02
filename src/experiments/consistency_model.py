from collections import defaultdict

import torch
from diffusers import LCMScheduler
from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("consistency_model")
class ConsistencyModelMethod(BaseMethod):
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
        self.guidance_scale = self.config.experiment_params.guidance_scale

    def run_experiment(self):
        self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)

        self.model.load_lora_weights(self.config.experiment_params.adapter_id)
        self.model.fuse_lora()

        batch_size = self.config.inference.get("batch_size", 1)

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        self.metric_dict = defaultdict(list)
        for idx_step, steps in enumerate(self.num_inference_steps):
            self.model.to(self.device)
            gen_images = self.generate(
                test_dataloader, steps, batch_size, guidance_scale=self.guidance_scale
            )
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
                name_table="Consistency model",
            )

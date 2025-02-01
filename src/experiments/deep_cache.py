from collections import defaultdict

import torch
from DeepCache import DeepCacheSDHelper
from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("deep_cache")
class DeepCacheMethod(BaseMethod):
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

        self.cache_interval = config.experiment_params.cache_interval
        self.cache_branch_id = config.experiment_params.get("cache_branch_id", 0)
        self.num_inference_steps = config.experiment_params.num_inference_steps

    def run_experiment(self):
        batch_size = self.config.inference.get("batch_size", 1)

        for cache_interval in self.cache_interval:
            helper = DeepCacheSDHelper(pipe=self.model)
            helper.set_params(
                cache_interval=cache_interval,
                cache_branch_id=self.cache_branch_id,
            )
            helper.enable()

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
                    name_images=f"Inference steps: {steps}, Cache interval: {cache_interval}",
                )

                self._update_metric_dict(steps)

                self.logger.log_metrics_into_table(
                    metrics=self.metric_dict,
                    name_table=f"Cache interval: {cache_interval}",
                )

        helper.disable()

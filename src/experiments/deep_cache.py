from collections import defaultdict

from DeepCache import DeepCacheSDHelper
from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("deep_cache")
class DeepCacheMethod(BaseMethod):
    def setup_exp_params(self):
        self.cache_interval = self.config.experiment_params.cache_interval
        self.cache_branch_id = self.config.experiment_params.get("cache_branch_id", 0)
        self.num_inference_steps = self.config.experiment_params.num_inference_steps

    def setup_scheduler(self):
        return None

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
                    name_images=f"{self.config.experiment_name}, Inference steps: {steps}, Cache interval: {cache_interval}",
                    name_table=f"{self.config.experiment_name}",
                    inference_step=steps,
                    additional_values={"Cache interval": cache_interval},
                )

            helper.disable()

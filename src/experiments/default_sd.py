from collections import defaultdict

from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("default")
class DefaultStableDiffusion(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps = self.config.experiment_params.num_inference_steps

    def setup_scheduler(self):
        return None

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
                name_table="Default stable diffusion",
                inference_step=steps,
            )

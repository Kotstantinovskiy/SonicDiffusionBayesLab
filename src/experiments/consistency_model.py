from collections import defaultdict

from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("consistency_model")
class ConsistencyModelMethod(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps = self.config.experiment_params.num_inference_steps
        self.guidance_scale = self.config.experiment_params.guidance_scale

        self.batch_size = self.config.inference.get("batch_size", 1)

    def run_experiment(self):
        # self.model.scheduler = LCMScheduler.from_config(self.model.scheduler.config)

        self.model.load_lora_weights(self.config.experiment_params.adapter_id)
        self.model.fuse_lora()

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.metric_dict = defaultdict(list)
        for idx_step, steps in enumerate(self.num_inference_steps):
            self.model.to(self.device)
            gen_images = self.generate(
                test_dataloader,
                steps,
                self.batch_size,
                guidance_scale=self.guidance_scale,
            )
            self.model.to("cpu")

            gen_dataloader = DataLoader(
                gen_images,
                batch_size=self.batch_size,
                shuffle=False,
            )

            # update metrics
            self.validate(
                test_dataloader,
                gen_dataloader,
                name_images=f"{self.config.experiment_name}, Inference steps: {steps}",
                name_table=f"{self.config.experiment_name}",
            )

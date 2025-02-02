from collections import defaultdict

import torch
from diffusers import DPMSolverMultistepScheduler
from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("dpm_solver")
class DPMSolverMethod(BaseMethod):
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

        self.algorithm_type = config.experiment_params.algorithm_type
        self.num_inference_steps = config.experiment_params.num_inference_steps
        self.num_train_timesteps = config.experiment_params.num_train_timesteps
        self.solver_order = config.experiment_params.solver_order
        self.final_sigmas_type = config.experiment_params.final_sigmas_type

    def run_experiment(self):
        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config,
        )

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
                name_images=f"Solver order: {self.solver_order}, Inference steps: {steps}",
                name_table="DPM Solver",
                inference_step=steps,
            )

from collections import defaultdict

from torch.utils.data import DataLoader

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("dpm_solver")
class DPMSolverMethod(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps = self.config.experiment_params.num_inference_steps
        self.solver_order = self.config.experiment_params.solver_order
        self.algorithm_type = self.config.experiment_params.algorithm_type

        self.batch_size = self.config.inference.get("batch_size", 1)

    def setup_scheduler(self, **kwargs):
        return super().setup_scheduler(solver_order=self.solver_order,
                                       algorithm_type=self.algorithm_type)

    def run_experiment(self):
        # self.model.scheduler = DPMSolverMultistepScheduler.from_config(
        #    self.model.scheduler.config,
        # )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.metric_dict = defaultdict(list)
        for idx_step, steps in enumerate(self.num_inference_steps):
            self.model.to(self.device)
            gen_images = self.generate(test_dataloader, steps, self.batch_size)
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
                name_images=f"{self.config.experiment_name}, Solver order: {self.solver_order}, Inference steps: {steps}",
                name_table=f"{self.config.experiment_name}",
            )

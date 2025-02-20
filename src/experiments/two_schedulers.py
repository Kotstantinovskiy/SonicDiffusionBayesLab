from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry, schedulers_registry


@methods_registry.add_to_registry("two_schedulers")
class TwoSchedulerMethod(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps_first = (
            self.config.experiment_params.num_inference_steps_first
        )
        self.num_inference_steps_second = (
            self.config.experiment_params.num_inference_steps_second
        )
        self.num_step_switch = self.config.experiment_params.num_step_switch
        self.solver_order = self.config.experiment_params.solver_order
        self.type_switch = self.config.experiment_params.type_switch

        self.batch_size = self.config.inference.get("batch_size", 1)

    def setup_scheduler(self):
        scheduler_first_name = self.config.scheduler.scheduler_first
        scheduler_second_name = self.config.scheduler.scheduler_second
        self.model.scheduler_first = schedulers_registry[
            scheduler_first_name
        ].from_config(self.model.scheduler.config)
        self.model.scheduler_second = schedulers_registry[
            scheduler_second_name
        ].from_config(self.model.scheduler.config)

    def generate(
        self,
        test_dataloader,
        num_inference_steps_first,
        num_inference_steps_second,
        num_step_switch,
        batch_size=1,
        guidance_scale=7.5,
    ):
        gen_images_list: list = []
        for idx, batch in enumerate(
            tqdm(
                test_dataloader,
                total=len(test_dataloader),
                desc="Experiment",
            )
        ):
            image_file, real_images, prompts = (
                batch["image_file"],
                batch["image"],
                batch["prompt"],
            )
            diffusion_gen_imgs, inference_time = self.model(
                prompts,
                guidance_scale=guidance_scale,
                generator=self.generator,
                num_inference_steps_first=num_inference_steps_first,
                num_inference_steps_second=num_inference_steps_second,
                num_step_switch=num_step_switch,
                type_switch=self.type_switch,
                output_type="pt",
            )
            diffusion_gen_imgs = diffusion_gen_imgs.images.cpu()

            gen_images = [
                diffusion_gen_imgs[dim_idx]
                for dim_idx in range(diffusion_gen_imgs.shape[0])
            ]
            gen_images_list.extend(gen_images)

            # update speed metrics
            self.time_metric.update(inference_time, batch_size)

        return gen_images_list

    def run_experiment(self):
        # self.model.scheduler_first = DPMSolverMultistepScheduler.from_config(
        #    self.model.scheduler.config,
        # )
        # self.model.scheduler_second = DPMSolverMultistepScheduler.from_config(
        #    self.model.scheduler.config,
        # )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.metric_dict = defaultdict(list)
        for (
            num_inference_steps_first,
            num_inference_steps_second,
            num_step_switch,
        ) in zip(
            self.num_inference_steps_first,
            self.num_inference_steps_second,
            self.num_step_switch,
        ):
            self.model.to(self.device)
            gen_images = self.generate(
                test_dataloader,
                num_inference_steps_first,
                num_inference_steps_second,
                num_step_switch,
                self.batch_size,
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
                name_images=f"{self.config.experiment_name}, Step first: {num_inference_steps_first}, Step second: {num_inference_steps_second}, Switch: {num_step_switch}",
                name_table="DPM and DDIM",
                additional_values={
                    "num_inference_steps_first": num_inference_steps_first,
                    "num_inference_steps_second": num_inference_steps_second,
                    "switch_step": num_step_switch,
                },
            )

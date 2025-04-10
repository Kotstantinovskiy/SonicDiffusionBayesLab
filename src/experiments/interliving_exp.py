from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry, schedulers_registry


@methods_registry.add_to_registry("interliving_schedulers")
class InterlivingSchedulerMethod(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps_first = (
            self.config.experiment_params.num_inference_steps_first
        )
        self.interliving_steps = self.config.experiment_params.interliving_steps

        self.first_order_solver = self.config.experiment_params.get('first_order_solver', '')
        self.second_order_solver = self.config.experiment_params.get('second_order_solver', '')
        
        self.first_algorithm_type = self.config.experiment_params.get('first_algorithm_type', '')
        self.second_algorithm_type = self.config.experiment_params.get('second_algorithm_type', '')
        self.first_final_sigmas_type = self.config.experiment_params.get('first_final_sigmas_type', '')
        self.second_final_sigmas_type = self.config.experiment_params.get('second_final_sigmas_type', '')

        self.batch_size = self.config.inference.get("batch_size", 1)

    def setup_scheduler(self):
        scheduler_first_name = self.config.scheduler.scheduler_main
        scheduler_second_name = self.config.scheduler.scheduler_inter
        self.model.scheduler_main = schedulers_registry[
            scheduler_first_name
        ].from_config(self.model.scheduler.config,
                      sovler_order=self.first_order_solver,
                      algorithm_type=self.first_algorithm_type,
                      final_sigmas_type=self.first_final_sigmas_type,)
        self.model.scheduler_inter = schedulers_registry[
            scheduler_second_name
        ].from_config(
            self.model.scheduler.config,
            sovler_order=self.second_order_solver,
            algorithm_type=self.second_algorithm_type,
            final_sigmas_type=self.second_final_sigmas_type,
        )

    def generate(
        self,
        test_dataloader,
        num_inference_steps,
        interliving_steps,
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
                num_inference_steps=num_inference_steps,
                interliving_steps=interliving_steps,
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
            interliving_steps,
        ) in zip(
            self.num_inference_steps_first,
            self.interliving_steps,
        ):
            self.model.to(self.device)
            gen_images = self.generate(
                test_dataloader=test_dataloader,
                num_inference_steps=num_inference_steps_first,
                interliving_steps=interliving_steps,
                batch_size=self.batch_size,
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
                name_images=f"{self.config.experiment_name}, Step main: {num_inference_steps_first}, Inter steps:{" ".join(interliving_steps)}",
                name_table=f"{self.config.experiment_name}",
                additional_values={
                    "num_inference_steps_main": num_inference_steps_first,
                    "num_inter_steps": " ".join(interliving_steps),
                },
            )

from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("skip_steps")
class SkipStepsMethod(BaseMethod):
    def setup_exp_params(self):
        self.skip_steps = self.config.experiment_params.skip_steps
        self.num_inference_steps = self.config.experiment_params.num_inference_steps
        self.solver_order = self.config.experiment_params.solver_order
        self.algorithm_type = self.config.experiment_params.algorithm_type
        self.final_sigmas_type = self.config.experiment_params.final_sigmas_type

        self.batch_size = self.config.inference.get("batch_size", 1)

    def setup_scheduler(self, **kwargs):
        return super().setup_scheduler(
            solver_order=self.solver_order,
            algorithm_type=self.algorithm_type,
            final_sigmas_type=self.final_sigmas_type,
        )

    def generate(
        self,
        test_dataloader,
        num_inference_steps,
        skip_steps,
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
            if self.config.inference.get("batch_count", None) is not None and idx >= self.config.inference.get("batch_count", None):
                break
            
            image_file, real_images, prompts = (
                batch["image_file"],
                batch["image"],
                batch["prompt"],
            )
            diffusion_gen_imgs, inference_time, x0_preds = self.model(
                prompts,
                guidance_scale=guidance_scale,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
                skip_timesteps=skip_steps,
                output_type="pt",
            )

            print("X0 preds", len(x0_preds))

            diffusion_gen_imgs = diffusion_gen_imgs.images.cpu()

            gen_images = [
                diffusion_gen_imgs[dim_idx]
                for dim_idx in range(diffusion_gen_imgs.shape[0])
            ]
            gen_images_list.extend(gen_images)

            # update speed metrics
            self.time_metric.update(inference_time, batch_size)

        return gen_images_list, x0_preds

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
            num_inference_steps,
            skip_steps,
        ) in zip(
            self.num_inference_steps,
            self.skip_steps,
        ):
            self.model.to(self.device)
            gen_images, x0_preds = self.generate(
                test_dataloader=test_dataloader,
                num_inference_steps=num_inference_steps,
                skip_steps=skip_steps,
                batch_size=self.batch_size,
            )
            self.model.to("cpu")

            gen_dataloader = DataLoader(
                gen_images,
                batch_size=self.batch_size,
                shuffle=False,
            )

            x0_preds_dataloader = None

            self.logger.log_batch_of_images(
                images=x0_preds,
                name_images="X0 preds",
            )

            # update metrics
            self.validate(
                test_dataloader,
                gen_dataloader,
                name_images=f"{self.config.experiment_name}, Step main: {num_inference_steps}, Skip steps:{' '.join(list(map(str, skip_steps)))}",
                name_table=f"{self.config.experiment_name}",
                additional_values={
                    "num_inference_steps": num_inference_steps,
                    "skip_steps": " ".join(list(map(str, skip_steps))),
                },
                x0_preds_dataloader=x0_preds_dataloader
                if self.config.experiment_params.get("use_x0", False)
                else None,
            )

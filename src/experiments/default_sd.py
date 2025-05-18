from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader 

from src.experiments.base_experiment import BaseMethod
from src.registry import methods_registry


@methods_registry.add_to_registry("default")
class DefaultStableDiffusion(BaseMethod):
    def setup_exp_params(self):
        self.num_inference_steps = self.config.experiment_params.num_inference_steps

    def setup_scheduler(self):
        return None
    
    def generate(
        self,
        test_dataloader,
        num_inference_steps,
        batch_size=1,
        guidance_scale=7.5,
    ):
        gen_images_list: list = []
        x0_preds_list: list = []
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
                output_type="pt",
            )

            x0_preds_list.extend(x0_preds)

            diffusion_gen_imgs = diffusion_gen_imgs.images.cpu()

            gen_images = [
                diffusion_gen_imgs[dim_idx]
                for dim_idx in range(diffusion_gen_imgs.shape[0])
            ]
            gen_images_list.extend(gen_images)

            # update speed metrics
            self.time_metric.update(inference_time, batch_size)

        return gen_images_list, x0_preds_list


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
            gen_images, x0_preds = self.generate(test_dataloader, steps, batch_size)
            self.model.to("cpu")

            gen_dataloader = DataLoader(
                gen_images,
                batch_size=batch_size,
                shuffle=False,
            )

            print(x0_preds)

            self.logger.log_batch_of_images(
                images=x0_preds,
                name_images=f"X0 preds {self.config.experiment_name}, Inference steps: {steps}",
            )

            # update metrics
            self.validate(
                test_dataloader,
                gen_dataloader,
                name_images=f"{self.config.experiment_name}, Inference steps: {steps}",
                name_table=f"{self.config.experiment_name}",
            )

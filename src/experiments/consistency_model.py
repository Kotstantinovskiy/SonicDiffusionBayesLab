from collections import defaultdict

import torch
from diffusers import LCMScheduler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
from src.experiments.base_experiment import BaseMethod
from src.loggers.wandb import Logger
from src.registry import methods_registry, metrics_registry, models_registry


@methods_registry.add_to_registry("consistency_model")
class ConsistencyModelMethod(BaseMethod):
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

        self.num_inference_steps = config.experiment_params.num_inference_steps
        self.num_train_timesteps = config.experiment_params.num_train_timesteps

    def run_experiment(self):
        self.model.scheduler = LCMScheduler(
            num_train_timesteps=self.num_train_timesteps,
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

            gen_images_list: list = []
            for idx, batch in enumerate(
                tqdm(
                    test_dataloader,
                    total=len(test_dataloader),
                    desc="DeepCache Experiment",
                )
            ):
                image_file, real_images, prompts = (
                    batch["image_file"],
                    batch["image"],
                    batch["prompt"],
                )
                diffusion_gen_imgs, inference_time = self.model(
                    prompts,
                    num_inference_steps=steps,
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

            self.model.to("cpu")

            gen_dataloader = DataLoader(
                gen_images_list,
                batch_size=batch_size,
                shuffle=False,
            )

            # update metrics
            for idx, (input_batch, gen_images) in tqdm(
                enumerate(zip(test_dataloader, gen_dataloader)),
                total=len(test_dataloader),
                desc="Calculating metrics...",
            ):
                image_file, real_images, prompts = (
                    batch["image_file"],
                    batch["image"],
                    batch["prompt"],
                )
                real_images = (real_images * 255).to(torch.uint8).cpu()
                gen_images = (gen_images * 255).to(torch.uint8).cpu()

                self.clip_score_gen_metric.update(gen_images, prompts)
                self.clip_score_real_metric.update(real_images, prompts)

                self.image_reward_metric.update(real_images, gen_images, prompts)

                self.fid_metric.update(gen_images, real=False)
                self.fid_metric.update(real_images, real=True)

                if idx % self.config.logger.log_images_step == 0:
                    self.logger.log_batch_of_images(
                        images=gen_images[:10],
                        name_images=f"Inference steps: {steps}",
                    )

            self._update_metric_dict(steps)

            self.logger.log_metrics_into_table(
                metrics=self.metric_dict,
                name_table="Consistency model",
            )

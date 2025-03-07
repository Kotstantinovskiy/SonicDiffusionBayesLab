from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
from src.loggers.wandb import Logger
from src.registry import metrics_registry, models_registry, schedulers_registry
from src.utils.model_utils import save_image, save_table, to_pil_image


class BaseMethod(ABC):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # setup generator
        self.setup_generator()

        # setup model
        self.setup_model()

        # setup schedulers
        self.setup_scheduler()

        # setup datasets
        self.setup_dataset()

        # metrics
        self.setup_metrics()

        # loggers
        self.setup_loggers()

        # setup experimant params
        self.setup_exp_params()

    @abstractmethod
    def run_experiment(self):
        pass

    def setup_exp_params(self):
        pass

    def setup_generator(self):
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.experiment.seed)

    def setup_model(self):
        model_name = self.config.model.model_name
        self.model = models_registry[model_name].from_pretrained(
            self.config.model.pretrained_model,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)

    def setup_scheduler(self):
        scheduler_name = self.config.scheduler.scheduler_name
        self.model.schedluers = schedulers_registry[scheduler_name].from_config(
            self.model.scheduler.config
        )

    def setup_dataset(self):
        self.dataset_test_dir = self.config.dataset.img_dataset
        self.prompts_test_file = self.config.dataset.prompts
        self.image_size = self.config.dataset.image_size

        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

        self.test_dataset = ImageDatasetWithPrompts(
            image_dir=self.dataset_test_dir,
            prompts_file=self.prompts_test_file,
            transform=transform,
        )

    def setup_metrics(self):
        self.metric_dict = defaultdict(list)

        self.clip_score_gen_metric = metrics_registry["clip_score"](
            model_name_or_path=self.config.quality_metrics.clip_score.model_name_or_path
        )

        self.image_reward_metric = metrics_registry["image_reward"](
            model_name=self.config.quality_metrics.image_reward.model_name,
            device=self.device,
        )

        self.fid_metric = metrics_registry["fid"](
            feature=self.config.quality_metrics.fid.feature,
            input_img_size=self.config.quality_metrics.fid.input_img_size,
            normalize=self.config.quality_metrics.fid.normalize,
        )

        self.time_metric = metrics_registry["time_metric"]()

    def setup_loggers(self):
        self.logger = Logger(
            config=OmegaConf.to_container(self.config, resolve=True),
            wandb_enable=self.config.logger.get("wandb_enable", True),
            project_name=self.config.logger.get("project_name", None),
            run_name=self.config.experiment_name,
            run_id=self.config.logger.get("run_id", None),
        )

    def generate(self, test_dataloader, steps, batch_size=1, guidance_scale=7.5):
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
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=self.generator,
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

    def validate(
        self,
        test_dataloader,
        gen_dataloader,
        name_images,
        name_table,
        additional_values: dict = None,
    ):
        self.clip_score_gen_metric.to(self.device)
        self.fid_metric.to(self.device)

        for idx, (input_batch, gen_images) in tqdm(
            enumerate(zip(test_dataloader, gen_dataloader)),
            total=len(test_dataloader),
            desc="Calculating metrics...",
        ):
            image_files, real_images, prompts = (
                input_batch["image_file"],
                input_batch["image"],
                input_batch["prompt"],
            )
            real_images = (real_images * 255).to(torch.uint8).cpu()
            gen_images = (gen_images * 255).to(torch.uint8).cpu()

            self.clip_score_gen_metric.update(gen_images.to(self.device), prompts)

            self.image_reward_metric.update(real_images, gen_images, prompts)

            self.fid_metric.update(gen_images.to(self.device), real=False)
            self.fid_metric.update(real_images.to(self.device), real=True)

            if idx % self.config.logger.log_images_step == 0:
                self.logger.log_batch_of_images(
                    images=gen_images[:16],
                    name_images=name_images,
                    captions=prompts[:16],
                )

            if self.config.logger.save:
                for image_file, gen_image in zip(image_files, gen_images.unbind(0)):
                    save_image(
                        self.config.logger.save_dir.format(
                            experiment=self.config.experiment_name,
                            args=name_images,
                        ),
                        image_file,
                        to_pil_image(gen_image),
                    )

        self.clip_score_gen_metric.to("cpu")
        self.fid_metric.to("cpu")

        if additional_values:
            for k, v in additional_values.items():
                self.metric_dict[k].append(v)

        self.metric_dict["nfe"].append(self.model.num_timesteps)
        self.metric_dict["clip_score_gen_image"].append(
            self.clip_score_gen_metric.compute().item()
        )

        self.metric_dict["image_reward"].append(
            self.image_reward_metric.compute().item()
        )

        self.metric_dict["fid"].append(self.fid_metric.compute().item())
        self.metric_dict["time_metric"].append(self.time_metric.compute().item())

        if self.config.logger.save:
            save_table(
                self.config.logger.save_dir.format(
                    experiment=self.config.experiment_name,
                    args=name_images,
                ),
                "metrics",
                pd.DataFrame.from_dict(self.metric_dict, orient="columns"),
            )

        self.logger.log_metrics_into_table(
            metrics=self.metric_dict,
            name_table=name_table,
        )

        self.fid_metric.reset()
        self.clip_score_gen_metric.reset()
        self.image_reward_metric.reset()
        self.time_metric.reset()

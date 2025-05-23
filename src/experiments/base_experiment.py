from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.dataset.dataset import ImageDatasetWithPrompts
from src.loggers.wandb import Logger
from src.registry import metrics_registry, models_registry, schedulers_registry
from src.utils.model_utils import save_image, save_table, to_pil_image


class BaseMethod(ABC):
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # setup experimant params
        self.setup_exp_params()

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
            timestamps=self.config.model.get("timestamps", None),
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)

    def setup_scheduler(self, **kwargs):
        scheduler_name = self.config.scheduler.scheduler_name

        self.model.scheduler = schedulers_registry[scheduler_name].from_config(
            self.model.scheduler.config,
            **kwargs,
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

    def generate(
        self,
        test_dataloader: DataLoader,
        steps: int,
        batch_size: int = 1,
        guidance_scale: float = 7.5,
    ) -> tuple[list, list]:
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

        return gen_images_list, x0_preds

    def validate(
        self,
        test_dataloader,
        gen_dataloader,
        name_images,
        name_table,
        additional_values: dict = None,
        x0_preds_dataloader: DataLoader | None = None,
    ):
        self.clip_score_gen_metric.to(self.device)
        self.fid_metric.to(self.device)

        if x0_preds_dataloader is not None:
            data_iter = zip(test_dataloader, gen_dataloader, x0_preds_dataloader)
        else:
            data_iter = zip(test_dataloader, gen_dataloader)

        for idx, batch in tqdm(
            enumerate(data_iter),
            total=len(test_dataloader),
            desc="Calculating metrics...",
        ):
            if x0_preds_dataloader is not None:
                input_batch, gen_images, x0_preds = batch
            else:
                input_batch, gen_images = batch
                x0_preds = None

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

            if idx % self.config.logger.get("log_images_step", 1) == 0:
                number_save_images = self.config.experiment.get("number_save_images", 8)

                self.logger.log_batch_of_images(
                    images=gen_images[:number_save_images],
                    name_images=name_images,
                    captions=prompts[:number_save_images],
                )

            if x0_preds and idx % self.config.logger.get("log_x0_step", 1) == 0:
                number_x0 = self.config.experiment.get("number_x0", 1)

                self.logger.log_batch_of_images(
                    images=x0_preds[:number_x0],
                    name_images=name_images,
                    captions=prompts[:number_x0],
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

    def collate_grid(self, batch):
        print(batch)
        # x0_preds = batch[0]
        grid_images = []
        for timestep_images in zip(*batch):
            images_stack = torch.stack(list(timestep_images))
            grid = make_grid(images_stack, nrow=8, normalize=True, padding=2)
            grid_images.append(grid)
        return grid_images

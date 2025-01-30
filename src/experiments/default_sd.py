from collections import defaultdict

import torch
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
from src.experiments.base_experiment import BaseMethod
from src.loggers.wandb import Logger
from src.registry import methods_registry, metrics_registry, models_registry


@methods_registry.add_to_registry("deep_cache")
class DeepCacheMethod(BaseMethod):
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

        self.num_steps = config.experiment_params.get("num_steps", 50)

    def setup_model(self):
        model_name = self.config.model.model_name
        self.model = models_registry[model_name].from_pretrained(
            self.config.model.pretrained_model
        )
        self.model.to(self.device)

    def setup_dataset(self):
        self.dataset_test_dir = self.config.dataset.img_dataset
        self.prompts_test_file = self.config.dataset.prompts
        self.image_size = self.config.dataset.get("image_size", 512)

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
        self.quality_metrics = list(self.config.quality_metrics)
        self.speed_metrics = list(self.config.speed_metrics)

        if len(self.quality_metrics) == 0 or len(self.speed_metrics) == 0:
            assert False, "Quality and Speed metrics should be provided"

        self.clip_score_metric = None
        if self.config.quality_metrics.get("clip_score", None):
            self.clip_score_metric = metrics_registry["clip_score"](
                model_name_or_path=self.config.quality_metrics.clip_score.model_name_or_path
            )

        self.image_reward_metric = None
        if self.config.quality_metrics.get("image_reward", None):
            self.image_reward_metric = metrics_registry["image_reward"](
                model_name=self.config.quality_metrics.image_reward.model_name,
                device=self.device,
            )

        self.fid_metric = None
        if self.config.quality_metrics.get("fid", None):
            self.fid_metric = metrics_registry["fid"](
                feature=self.config.quality_metrics.fid.feature,
                input_img_size=self.config.quality_metrics.fid.input_img_size,
                normalize=self.config.quality_metrics.fid.normalize,
            )

        self.time_metric = None
        if self.config.quality_metrics.get("time_metric", None):
            self.time_metric = metrics_registry["time_metric"]()

    def setup_loggers(self):
        self.logger = Logger(
            config=OmegaConf.to_container(self.config, resolve=True),
            wandb_enable=self.config.logger.get("wandb_enable", True),
            project_name=self.config.logger.get("project_name", None),
            run_name=self.config.experiment_name,
            run_id=self.config.logger.get("run_id", None),
        )

    def run_experiment(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.inference.get("batch_size", 1),
            shuffle=False,
        )

        pred_images_list: list = []
        for idx, batch in enumerate(
            tqdm(
                test_dataloader, total=len(test_dataloader), desc="DeepCache Experiment"
            )
        ):
            real_images, prompts = batch["image"], batch["prompt"]
            diffusion_pred_imgs, inference_time = self.model(
                prompts, num_inference_steps=self.num_steps, output_type="np"
            )
            pred_images = diffusion_pred_imgs.images[0]
            pred_images_list.extend(pred_images)

            # update speed metrics
            if self.time_metric:
                self.time_metric.update(inference_time)

        pred_dataloader = DataLoader(
            pred_images_list,
            batch_size=self.config.inference.get("batch_size", 1),
            shuffle=False,
        )

        # update metrics
        for input_batch, output_images in tqdm(
            zip(test_dataloader, pred_dataloader),
            total=len(test_dataloader),
            desc="Calculating metrics...",
        ):
            real_images, prompts = input_batch["image"], input_batch["prompt"]

            if self.clip_score_metric:
                self.clip_score_metric.update(output_images, prompts)

            if self.image_reward_metric:
                self.image_reward_metric.update(output_images, prompts)

            if self.fid_metric:
                self.fid_metric.update(output_images, [0] * len(output_images))
                self.fid_metric.update(real_images, [1] * len(real_images))

        metric_dict = defaultdict(list)
        if self.clip_score_metric:
            metric_dict["clip_score"].append(self.clip_score_metric.compute())

        if self.image_reward_metric:
            metric_dict["image_reward"].append(self.image_reward_metric.compute())

        if self.fid_metric:
            metric_dict["fid"].append(self.fid_metric.compute())

        if self.time_metric:
            metric_dict["time_metric"].append(self.time_metric.compute())

        self.logger.log_metrics_into_table(metric_dict)

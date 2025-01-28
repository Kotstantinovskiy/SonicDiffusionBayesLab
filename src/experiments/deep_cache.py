from collections import defaultdict

import torch
from base_experiment import BaseMethod
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
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

        self.cache_interval = config.deepcache_params.get("cache_interval", 1)
        self.cache_branch_id = config.deepcache_params.get("cache_branch_id", 0)

    def setup_model(self):
        model_name = self.config.model_name
        self.model = models_registry[model_name].from_pretrained(model_name)
        self.model.to(self.device)

    def setup_dataset(self):
        self.dataset_test_dir = self.config.dataset.img_dataset
        self.prompts_test_file = self.config.dataset.prompts

        self.test_dataset = ImageDatasetWithPrompts(
            image_dir=self.dataset_test_dir, text_prompts_json=self.prompts_test_file
        )

    def setup_metrics(self):
        self.quality_metrics = self.config.quality_metrics
        self.speed_metrics = self.config.speed_metrics

        if len(self.quality_metrics) == 0 or len(self.speed_metrics) == 0:
            assert False, "Quality and Speed metrics should be provided"

        self.clip_score_metric = None
        if "clip_score" in self.config.quality_metrics:
            self.clip_score_metric = metrics_registry["clip_score"](
                model_name_or_path=self.config.quality_metrics.clip_score.model_name_or_path
            )

        self.image_reward_metric = None
        if "image_reward" in self.config.quality_metrics:
            self.image_reward_metric = metrics_registry["image_reward"](
                model_name=self.config.quality_metrics.image_reward.model_name,
                device=self.device,
            )

        self.fid_metric = None
        if "fid" in self.config.quality_metrics:
            self.fid_metric = metrics_registry["fid"](
                feature=self.config.quality_metrics.fid.feature,
                input_img_size=self.config.quality_metrics.fid.input_img_size,
                normalize=self.config.quality_metrics.fid.normalize,
            )

        self.time_metric = None
        if "time_metric" in self.config.speed_metrics:
            self.time_metric = metrics_registry["time_metric"]()

    def setup_loggers(self):
        self.logger = Logger(
            config=self.config,
            wandb_enable=self.config.logger.get("wandb_enable", True),
            project_name=self.config.logger.get("project_name", None),
            run_name=self.config.experiment_name,
            run_id=self.config.logger.get("run_id", None),
        )

    def run_experiment(self):
        helper = DeepCacheSDHelper(pipe=self.model)
        helper.set_params(
            cache_interval=self.cache_interval,
            cache_branch_id=self.cache_branch_id,
        )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.inference.get("batch_size", 1),
            shuffle=False,
        )

        helper.enable()

        for idx, batch in enumerate(
            tqdm(
                test_dataloader, total=len(test_dataloader), desc="DeepCache Experiment"
            )
        ):
            real_images, prompts = batch["image"], batch["prompt"]
            inference_time, diffusion_pred_imgs = self.model(prompts, output_type="pil")
            pred_images = diffusion_pred_imgs.images[0]

            # update metrics
            if self.clip_score_metric:
                self.clip_score_metric.update(pred_images, prompts)

            if self.image_reward_metric:
                self.image_reward_metric.update(pred_images, prompts)

            if self.fid_metric:
                self.fid_metric.update(pred_images, [0] * len(pred_images))
                self.fid_metric.update(real_images, [1] * len(real_images))

            # update speed metrics
            if self.time_metric:
                self.time_metric.update(inference_time)

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

        helper.disable()

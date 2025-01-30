from collections import defaultdict

import torch
from DeepCache import DeepCacheSDHelper
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

        self.cache_interval = config.experiment_params.cache_interval
        self.cache_branch_id = config.experiment_params.get("cache_branch_id", 0)
        self.num_inference_steps = config.experiment_params.num_inference_steps

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
        self.metric_dict = defaultdict(list)

        self.clip_score_gen_metric = metrics_registry["clip_score"](
            model_name_or_path=self.config.quality_metrics.clip_score.model_name_or_path
        )
        self.clip_score_real_metric = metrics_registry["clip_score"](
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

    def _update_metric_dict(self, inference_step):
        self.metric_dict["nfe"].append(inference_step)
        self.metric_dict["clip_score_gen_image"].append(
            self.clip_score_gen_metric.compute().item()
        )
        self.metric_dict["clip_score_real_image"].append(
            self.clip_score_real_metric.compute().item()
        )

        self.metric_dict["image_reward"].append(
            self.image_reward_metric.compute().item()
        )

        self.metric_dict["fid"].append(self.fid_metric.compute().item())
        self.metric_dict["time_metric"].append(self.time_metric.compute().item())

    def run_experiment(self):
        for cache_interval in self.cache_interval:
            helper = DeepCacheSDHelper(pipe=self.model)
            helper.set_params(
                cache_interval=cache_interval,
                cache_branch_id=self.cache_branch_id,
            )

            test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.config.inference.get("batch_size", 1),
                shuffle=False,
            )

            self.metric_dict = defaultdict(list)
            for idx_step, steps in enumerate(self.num_inference_steps):
                self.model.to(self.device)

                helper.enable()

                gen_images_list: list = []
                for idx, batch in enumerate(
                    tqdm(
                        test_dataloader,
                        total=len(test_dataloader),
                        desc="DeepCache Experiment",
                    )
                ):
                    real_images, prompts = batch["image"], batch["prompt"]
                    diffusion_gen_imgs, inference_time = self.model(
                        prompts,
                        num_inference_steps=steps,
                        output_type="pt",
                    )
                    diffusion_gen_imgs = diffusion_gen_imgs.images

                    gen_images = [
                        diffusion_gen_imgs[dim_idx]
                        for dim_idx in range(diffusion_gen_imgs.shape[0])
                    ]
                    gen_images_list.extend(gen_images)

                    # update speed metrics
                    if self.time_metric:
                        self.time_metric.update(inference_time)

                self.model.to("cpu")

                gen_dataloader = DataLoader(
                    gen_images_list,
                    batch_size=self.config.inference.get("batch_size", 1),
                    shuffle=False,
                )

                # update metrics
                for idx, (input_batch, gen_images) in tqdm(
                    enumerate(zip(test_dataloader, gen_dataloader)),
                    total=len(test_dataloader),
                    desc="Calculating metrics...",
                ):
                    real_images, prompts = input_batch["image"], input_batch["prompt"]
                    real_images = (real_images * 255).to(torch.uint8).cpu()
                    gen_images = (gen_images * 255).to(torch.uint8).cpu()

                    self.clip_score_gen_metric.update(gen_images, prompts)
                    self.clip_score_real_metric.update(real_images, prompts)
                    self.image_reward_metric.update(gen_images, prompts)

                    self.fid_metric.update(gen_images, real=False)
                    self.fid_metric.update(real_images, real=True)

                    if idx % self.config.logger.log_images_step == 0:
                        self.logger.log_batch_of_images(
                            images=gen_images[:10],
                            name_images="Generated images",
                            step=idx,
                        )

                self._update_metric_dict(steps)

                self.logger.log_metrics_into_table(
                    metrics=self.metric_dict,
                    name_table=f"Cache interval: {cache_interval}",
                )

                helper.disable()

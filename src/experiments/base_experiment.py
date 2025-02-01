from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts
from src.loggers.wandb import Logger
from src.registry import metrics_registry, models_registry
from utils.model_utils import save_image, to_pil_image


class BaseMethod(ABC):
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

    @abstractmethod
    def run_experiment(self):
        pass

    def setup_model(self):
        model_name = self.config.model.model_name
        self.model = models_registry[model_name].from_pretrained(
            self.config.model.pretrained_model,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16,
        )
        self.model.to(self.device)

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

    def generate(self, test_dataloader, steps, batch_size=1):
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

        return gen_images_list

    def validate(self, test_dataloader, gen_dataloader, name_images):
        for idx, (input_batch, gen_images) in tqdm(
            enumerate(zip(test_dataloader, gen_dataloader)),
            total=len(test_dataloader),
            desc="Calculating metrics...",
        ):
            image_file, real_images, prompts = (
                input_batch["image_file"],
                input_batch["image"],
                input_batch["prompt"],
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
                    images=gen_images[:16],
                    name_images=name_images,
                )

            if self.config.logger.save_images:
                for gen_image in gen_images.unbind(0):
                    save_image(
                        self.config.logger.save_images_dir.format(
                            experiment=self.config.experiment_name
                        ),
                        image_file,
                        to_pil_image(gen_image),
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

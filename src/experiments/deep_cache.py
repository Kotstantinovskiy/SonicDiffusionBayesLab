import time

from base_experiment import BaseMethod
from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader

from src.dataset.dataset import ImageDatasetWithPrompts
from src.registry import methods_registry, metrics_registry


@methods_registry.add_to_registry("deep_cache")
class DeepCacheMethod(BaseMethod):
    def __init__(self, config):
        self.config = config

        # setup model
        self.setup_model()

        # setup datasets
        self.setup_dataset()

        # metrics
        self.setup_metrics()

        self.cache_interval = config.deepcache_params.get("cache_interval", 1)
        self.cache_branch_id = config.deepcache_params.get("cache_branch_id", 0)

    def setup_model(self):
        model_name = self.config.model_name
        self.model = StableDiffusionPipeline.from_pretrained(model_name)

    def setup_dataset(self):
        self.dataset_train_dir = self.config.dataset.train.img_dataset
        self.prompts_train_file = self.config.dataset.train.prompts

        self.dataset_val_dir = self.config.dataset.val.img_dataset
        self.prompts_val_file = self.config.dataset.val.prompts

        self.dataset_test_dir = self.config.dataset.test.img_dataset
        self.prompts_test_file = self.config.dataset.test.prompts

        self.train_dataset = ImageDatasetWithPrompts(
            image_dir=self.dataset_train_dir, text_prompts_json=self.prompts_train_file
        )

        self.val_dataset = ImageDatasetWithPrompts(
            image_dir=self.dataset_val_dir, text_prompts_json=self.prompts_val_file
        )

        self.test_dataset = ImageDatasetWithPrompts(
            image_dir=self.dataset_test_dir, text_prompts_json=self.prompts_test_file
        )

    def setup_metrics(self):
        self.quality_metrics = self.config.quality_metrics
        self.speed_metrics = self.config.speed_metrics

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
                device=self.config.device,
            )

        self.fid_metric = None
        if self.config.quality_metrics.get("fid", None):
            self.fid_metric = metrics_registry["fid"](
                feature=self.config.quality_metrics.fid.feature,
                input_img_size=self.config.quality_metrics.fid.input_img_size,
                normalize=self.config.quality_metrics.fid.normalize,
            )

        self.time_metric = None
        if self.config.speed_metrics.get("time_metric", None):
            self.time_metric = metrics_registry["time_metric"]()

    def run_experiment(self):
        helper = DeepCacheSDHelper(pipe=self.model)
        helper.set_params(
            cache_interval=self.cache_interval,
            cache_branch_id=self.cache_branch_id,
        )

        helper.enable()

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.get("batch_size", 1),
            shuffle=False,
        )

        for idx, batch in enumerate(dataloader):
            start_inference_time = time.time()
            pil_images, prompts = batch["image"], batch["prompt"]
            pred_imgs = self.model(prompts, output_type="pil").images[0]
            end_inference_time = time.time()

            # update metrics
            if self.clip_score_metric:
                self.clip_score_metric.update(pred_imgs, prompts)

            if self.image_reward_metric:
                self.image_reward_metric.update(pred_imgs, prompts)

            if self.fid_metric:
                self.fid_metric.update(pred_imgs, prompts)

            # update speed metrics
            if self.fid_metric:
                self.time_metric.update(end_inference_time - start_inference_time)

        helper.disable()

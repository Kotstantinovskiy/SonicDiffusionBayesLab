from abc import ABC, abstractmethod
from typing import Any

import ImageReward as RM
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from typing_extensions import Literal

from src.metrics.aethetic_score_model import AethteticScoreMLP
from src.registry import metrics_registry
from src.utils.model_utils import to_pil_image


class CustomMetric(ABC):
    @abstractmethod
    def calc_metric(self) -> float:
        pass


@metrics_registry.add_to_registry("clip_score")
class ClipScoreMetric(CLIPScore, CustomMetric):
    def calc_metric(
        self, data: list[Image.Image], prompts: list[str], batch_size: int = 4
    ) -> float:
        data_tensors = [pil_to_tensor(img) for img in data]
        dataloader_imgs = DataLoader(data_tensors, batch_size=batch_size, shuffle=False)
        dataloader_prompts = DataLoader(prompts, batch_size=batch_size, shuffle=False)

        for imgs_batch, prompts_batch in tqdm(
            zip(dataloader_imgs, dataloader_prompts),
            desc="Calculating clip score",
            total=len(dataloader_imgs),
        ):
            self.update(images=imgs_batch, text=prompts_batch)

        return self.compute().item()


@metrics_registry.add_to_registry("image_reward")
class RewardModel(Metric, CustomMetric):
    def __init__(
        self,
        model_name: Literal["ImageReward-v1.0"] = "ImageReward-v1.0",
        device: str = "cpu",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.rm_model = RM.load(name=model_name, device=device)
        self.add_state("reward_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, prompts: list[str] | str) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        return prompts

    def update(
        self,
        real_imgs: list[Image.Image],
        gen_imgs: list[Image.Image],
        prompts: list[str] | str,
    ) -> None:
        prompts = self._input_format(prompts)
        real_imgs = [to_pil_image(img) for img in real_imgs]
        gen_imgs = [to_pil_image(img) for img in gen_imgs]
        if len(real_imgs) != len(prompts):
            raise ValueError("Imgs and prompts must have the same size")

        for prompt, real_img, gen_img in zip(prompts, real_imgs, gen_imgs):
            idxs, reward = self.rm_model.inference_rank(prompt, [real_img, gen_img])
            self.reward_sum += torch.tensor(int(reward[0] <= reward[1]))
            self.total += 1

    def compute(self) -> torch.Tensor:
        return self.reward_sum.float() / self.total

    def calc_metric(
        self,
        imgs: list[Image.Image],
        prompts: list[str],
    ) -> float:
        _, rewards = self.rm_model.inference_rank(prompts, imgs)[1]
        self.reward_sum = torch.tensor(sum(rewards))
        self.total = torch.tensor(len(imgs))

        return self.reward_sum / self.total


@metrics_registry.add_to_registry("fid")
class FID(FrechetInceptionDistance, CustomMetric):
    def calc_metric(
        self,
        imgs: list[Image.Image],
        reals: list[bool],
    ) -> float:
        imgs_tensors = [pil_to_tensor(img) for img in imgs]

        for img_tensor, real in tqdm(
            zip(imgs_tensors, reals), desc="Calculating FID", total=len(imgs_tensors)
        ):
            self.update(imgs=img_tensor, real=real)

        return self.compute().item()


@metrics_registry.add_to_registry("time_metric")
class TimeMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, time: float, batch_size: int) -> None:
        self.time += torch.tensor(time)
        self.total += torch.tensor(batch_size)

    def compute(self) -> float:
        return self.time / self.total

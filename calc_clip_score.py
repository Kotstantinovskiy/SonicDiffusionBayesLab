import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms
from tqdm import tqdm

from src.dataset.dataset import ImageDatasetWithPrompts


def calc_clip_score(
    test_dataloader: DataLoader,
    model_name_or_path: str = "openai/clip-vit-base-patch16",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> float:
    clip_score_metric = CLIPScore(model_name_or_path=model_name_or_path).to(device)

    for batch in tqdm(
        test_dataloader,
        total=len(test_dataloader),
        desc="Calculating CLIP score",
    ):
        image_file, real_images, prompts = (
            batch["image_file"],
            batch["image"],
            batch["prompt"],
        )

        # Move tensors to the correct device
        real_images = real_images.to(device)

        clip_score_metric.update(images=real_images, text=prompts)

    return clip_score_metric.compute().item()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate CLIP score for images and prompts"
    )
    parser.add_argument(
        "--folder_path", type=str, help="Path to folder containing images"
    )
    parser.add_argument(
        "--prompts_file", type=str, help="JSON file containing prompts for images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP model name or path to use for scoring",
    )
    args = parser.parse_args()

    if not args.folder_path or not os.path.isdir(args.folder_path):
        raise ValueError("Please provide a valid folder path containing images")

    if not args.prompts_file or not os.path.isfile(args.prompts_file):
        raise ValueError("Please provide a valid JSON file containing prompts")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create a dataset with prompts for images
    dataset = ImageDatasetWithPrompts(
        image_dir=args.folder_path,
        prompts_file=args.prompts_file,
        transform=transform,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Calculate CLIP score
    score = calc_clip_score(
        test_dataloader,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
    )

    print(f"CLIP Score: {score}")

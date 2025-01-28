import json
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDatasetWithPrompts(Dataset):
    def __init__(self, image_dir, prompts_file, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            prompts_file (dict): Json of text prompts corresponding to each image.
            transform (callable, optional): Optional transform to be applied on an image.
        """

        self.image_dir = image_dir
        self.prompts_file = prompts_file
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f))
        ]

        with open(self.prompts_file, "r") as f:
            self.prompts_json = json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img_name = os.path.join(self.image_dir, image_file)
        image = Image.open(img_name).convert("RGB")
        text_prompt = self.prompts_json[image_file]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "prompt": text_prompt}

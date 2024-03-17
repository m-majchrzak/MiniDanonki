from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tools.logger import get_logger
from tools.path_organizer import PathOrganizer


class OurDataset(Dataset):

    def __init__(self, name: str, transform: transforms.Compose | None = None, prefix: str | None = None) -> None:
        self.transform = transform
        self.name = name
        self.logger = get_logger("OurDataset")
        self.path_organizer = PathOrganizer(prefix)

        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self) -> None:
        all_images = os.listdir(self.path_organizer.get_images_dir())
        for image_name in all_images:
            repr_path = self.path_organizer.get_image_representation_path(image_name)
            if repr_path:
                #self.logger.info(f"Loading image {image_name} with repr: {repr_path}")
                img_path = self.path_organizer.get_image_path(image_name)
                self.images.append(img_path)
                self.labels.append(repr_path)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        label_np = np.load(label_path)["representation"]
        #label_np.astype(np.float32)
        label = torch.tensor(label_np, dtype=torch.float32)
        image = Image.open(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def get_dataloader_standard(self, batch_size: int = 1) -> DataLoader:
        return DataLoader(self, batch_size=batch_size)

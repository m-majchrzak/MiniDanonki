from __future__ import annotations

from torch.utils.data import DataLoader

from tools.logger import get_logger
from tools.path_organizer import PathOrganizer


class OurDataset:

    def __init__(self, name: str, prefix: str | None = None) -> None:
        self.name = name
        self.logger = get_logger("OurDataset")
        self.path_organizer = PathOrganizer(prefix)

    def __len__(self) -> int:
        pass  # TODO

    def get_dataloader_standard(self, batch_size: int = 1) -> DataLoader:
        pass  # TODO






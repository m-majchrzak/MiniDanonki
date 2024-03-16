from __future__ import annotations

import os
import pathlib


class PathOrganizer:
    """
    Class for organizing paths to project's resources
    """

    def __init__(self, prefix: str | None = None) -> None:
        self.prefix = prefix or PathOrganizer.get_root()

    @staticmethod
    def get_root() -> str:
        """
        :return: path to repo root
        """
        return str(pathlib.Path(__file__).parent.parent.absolute())

    def get_dataset_path(self, name: str) -> str:
        return os.path.join(self.prefix, "data", "dataset", name)

    def get_onnx_model_path(self, model_name: str, tag: str, epoch: int) -> str:
        """
        :return: path where segmentation model should be copied
        """
        return os.path.join(self.prefix, "data", "models", model_name, tag, f"{epoch}.onnx")

    def get_jsonl_metrics_path(self, model_name: str, tag: str) -> str:
        """
        :return: path where segmentation model should be copied
        """
        return os.path.join(self.prefix, "data", "models", model_name, tag, f"metrics.jsonl")

from __future__ import annotations

import os
import pathlib

from tools.fs_tools import FsTools


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

    def get_images_dir(self) -> str:
        return os.path.join(self.prefix, "modelstealing", "data", "labels_ids_png")

    def get_image_path(self, image_name: str) -> str:
        return os.path.join(self.prefix, "modelstealing", "data", "labels_ids_png", image_name)

    def get_image_representations_dir(self) -> str:
        return os.path.join(self.prefix, "modelstealing", "representations")

    def get_image_representation_path(self, image_name: str) -> str | None:
        dir_name = image_name.split(".")[0]
        try:
            image_representations = os.listdir(os.path.join(self.get_image_representations_dir(), dir_name))
        except FileNotFoundError:
            return None
        if not image_representations:
            return None
        chosen_repr_filename = list(sorted(image_representations))[0]
        return os.path.join(self.get_image_representations_dir(), dir_name, chosen_repr_filename)

from __future__ import annotations

import os
import pathlib


class FsTools:
    """filesystem tools"""

    @staticmethod
    def mkdir(path: str | os.PathLike) -> None:
        """
        Creates directory
        :param path: path to directory
        """
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def ensure_dir(path: str) -> None:
        """
        Creates parent directories for a file
        :param path: path to file
        """
        FsTools.mkdir(pathlib.Path(path).parent)

    @staticmethod
    def rm_file(path: str) -> None:
        """
        Removes file
        :param path: path to file
        """
        if os.path.isfile(path):
            os.remove(path)

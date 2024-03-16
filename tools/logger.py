from __future__ import annotations

import logging

from tools.fs_tools import FsTools

STDERR_VERBOSITY = logging.INFO


def set_loggers_stderr_verbosity(new_level: int | str) -> None:
    """
    :param new_level: new verbosity that will be set for STDERR handler for loggers,
        examples: 'logging.DEBUG', '0',
        lower the verbosity - more messages
    """
    global STDERR_VERBOSITY
    STDERR_VERBOSITY = new_level


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.Logger(logger_name)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(STDERR_VERBOSITY)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.debug(f"Logger {logger_name} initialized.")
    return logger


def add_file_handler(logger: logging.Logger, filepath: str, verbosity: str | int = logging.INFO) -> None:
    """
    :param logger: logger to whom handler will be added
    :param filepath: path to file to save log
    :param verbosity: verbosity of the handler
    """
    FsTools.ensure_dir(filepath)
    handler = logging.FileHandler(filepath)
    handler.setLevel(verbosity)
    logger.addHandler(handler)
    logger.debug(f"Added file handler. Logs now will be written to file {filepath}.")

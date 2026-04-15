#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup logging class for CLI """

import logging
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Iterable, Sized

from .cli_utils import format_spend_time, file_duration


class Logger:
    def __init__(self, name: str, output_dir: Path, verbose=False):
        # Create a logger
        self.name = name
        self.handlers = []
        self.output_dir: Path = output_dir
        self.verbose: bool = verbose
        self.logger = self._build_logger()

    def _build_logger(self):
        _logger = logging.getLogger(self.name)
        _logger.setLevel(logging.DEBUG)

        file_handler = self.get_file_handler(self.output_dir / "transcribe.log")
        _logger.addHandler(file_handler)

        # Add handlers to the logger
        if self.verbose:
            stream = self.get_stdout_handler()
            _logger.addHandler(stream)

        return _logger

    @staticmethod
    def get_stdout_handler():
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stdout_handler.setFormatter(stdout_formatter)
        return stdout_handler

    @staticmethod
    def get_file_handler(filename: Path):
        file_handler = logging.FileHandler(filename)
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        return file_handler

    def flush_stdout(self):
        if self.verbose:
            sys.stdout.flush()

    def log_model_loading(self, model_name: str, start_time: int, end_time: int):
        self.logger.debug(
            "Loading %s to %s", model_name, format_spend_time(start_time, end_time)
        )
        self.flush_stdout()

    def log_threads(self, threads: int):
        self.logger.debug("Threads initialised: %s", threads)
        self.flush_stdout()

    def log_processing(self, files: Sized):
        self.logger.debug("Processing #%s..", len(files))
        self.flush_stdout()

    def log_file_start(self, file: Path, device) -> float:
        duration: float = file_duration(file)
        self.logger.debug(
            "Starting %s duration: %d seconds on device: %s",
            file.name,
            duration,
            device,
        )
        self.flush_stdout()
        return duration

    def log_file_end(self, file: Path, start_time: int, end_time: int):
        self.logger.debug(
            "Processed %s it took %s",
            file.name,
            format_spend_time(start_time, end_time),
        )
        self.flush_stdout()

    def log_finished(self, filename: str):
        self.logger.debug("%s is finished", filename)
        self.flush_stdout()

    def get_logger(self):
        return self.logger


if __name__ == "__main__":
    import time

    logPath = Path("./outd")
    fileName = "udput"
    logPath.mkdir(exist_ok=True)
    logger = Logger(__name__, logPath, verbose=True)

    # Log messages
    start_time2 = perf_counter_ns()
    time.sleep(12)
    logger.log_model_loading("This is a debug message.", start_time2, perf_counter_ns())

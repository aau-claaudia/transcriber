#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """
from . import writers

from pathlib import Path
from whisperx import utils
from typing import TextIO

WRITERS = {
    name.strip("Write").lower(): cls
    for name, cls in writers.__dict__.items()
    if isinstance(cls, type) and name.startswith("Write")
}
OFFICIAL_WRITERS = {
    name.strip("Write").lower(): cls
    for name, cls in utils.__dict__.items()
    if isinstance(cls, type) and name.startswith("Write")
}

WRITERS.update(OFFICIAL_WRITERS)


def get_writer(output_format: str, output_dir: str | Path):
    if output_format == "all":
        all_writers = [writer(output_dir) for writer in WRITERS.values()]

        def write_all(result: dict, file: TextIO, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all

    return WRITERS[output_format](output_dir)

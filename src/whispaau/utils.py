#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """
from _csv import Writer

from . import writers

from pathlib import Path
from whisperx import utils
from typing import TextIO

WRITERS = {}

# this set of writers contains our own customized writers (csv, dote.json and docx)
CUSTOMIZED_WRITERS = {
    name.strip("Write").lower(): cls
    for name, cls in writers.__dict__.items()
    if isinstance(cls, type) and name.startswith("Write")
}
OFFICIAL_WRITERS = {
    name.strip("Write").lower(): cls
    for name, cls in utils.__dict__.items()
    if isinstance(cls, type) and name.startswith("Write")
}
MERGED_SPEAKERS_WRITERS = {
    name.strip("Write").lower(): cls
    for name, cls in utils.__dict__.items()
    if isinstance(cls, type) and (name.startswith("WriteSRT") or name.startswith("WriteVTT"))
}
# this set of writers contains our own customized writers and the official srt and vtt writers
MERGED_SPEAKERS_WRITERS.update(CUSTOMIZED_WRITERS)

# this set contains the customized and official writers
WRITERS.update(CUSTOMIZED_WRITERS)
WRITERS.update(OFFICIAL_WRITERS)


def get_writer(output_format: str, output_dir: str | Path, writer_list):
    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writer_list.values()]

        def write_all(result: dict, file: TextIO, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all
    # check if the selected format is one of the available formats, when requesting e.g. txt which is not in merge set
    if output_format in writer_list:
        return writer_list[output_format](output_dir)
    else:
        return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" """
from _csv import Writer

from . import writers

from pathlib import Path
from whisperx import utils
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
from typing import TextIO
from collections import OrderedDict

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

# Function to move the csv writer to the end
def move_csv_to_end(dictionary):
    if 'csv' in dictionary:
        dictionary.move_to_end('csv')
    return dictionary
# sorting the writer dictionaries so that the CSV writer runs last. The data is cleaned before writing CSV to avoid errors.
# see https://github.com/aau-claaudia/transcriber/issues/20
MERGED_SPEAKERS_WRITERS = move_csv_to_end(OrderedDict(sorted(MERGED_SPEAKERS_WRITERS.items())))
WRITERS = move_csv_to_end(OrderedDict(sorted(WRITERS.items())))

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

def is_speaker_diarization_supported(language: str) -> bool:
    return (language in DEFAULT_ALIGN_MODELS_HF) or (language in DEFAULT_ALIGN_MODELS_TORCH)
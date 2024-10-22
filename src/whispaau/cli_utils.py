#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utilities for AAU Transcription CLI """

from pathlib import Path
from typing import Optional, Any
import argparse
import ffmpeg
from datetime import datetime
from whisperx.utils import optional_int


def get_directory(input_dir: str):
    files = [
        p.resolve()
        for p in Path(input_dir).glob("**/*")
        if p.suffix.lower() in {".mp3", ".mp4", ".m4a", ".wav", ".mpg"}
    ]
    return files


class FileWrongFormat(Exception):
    pass


def file_duration(file_path: Path) -> float:
    info = ffmpeg.probe(file_path)
    return float(info["format"]["duration"])


def format_spend_time(start: int, end: int) -> str:
    spend_time = (end - start) / 1e9
    dt = datetime.utcfromtimestamp(spend_time)
    return dt.strftime("%H:%M:%S.%f")


def aggregate_paths(file_paths: list[Path], directory_paths: list[Path]) -> set[Path]:
    _files = (
        {file.resolve() for file in file_paths if file.exists()}
        if file_paths
        else set()
    )
    _directories = (
        {directory for directory in directory_paths if directory.exists()}
        if directory_paths
        else set()
    )
    return _files.union(_directories)


def parse_arguments(args: Optional[list[Any]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="large", type=str, help="what model is used?"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macos gpu training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables cuda training"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print info to screen"
    )
    parser.add_argument(
        "-d",
        "--input_dir",
        type=get_directory,
        # required=True,
        help="directory of multiple files (mp3, m4a, mp4, wav) for transcribing",
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=False,
        help="file for transcribring",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("./out"),
        help="file for transcribring",
    )
    parser.add_argument(
        "-la",
        "--language",
        type=str,
        required=False,
        help="language of audio, not set let whisper guess",
    )
    parser.add_argument(
        "-l", "--logging", action="store_true", default=False, help="create log file"
    )
    parser.add_argument(
        "--threads",
        type=optional_int,
        default=0,
        help="number of threads used by torch for cpu inference",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="all",
        help="what output format do you want?",
    )
    parser.add_argument("--prompt", type=str, default=[], nargs="+")
    parser.add_argument(
        "--archive_password",
        type=str,
        default=None,
        help="password for encrypted zip package?",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="jobname",
        help="what is the job name?",
    )

    parser.add_argument(
        "--min_speakers",
        type=optional_int,
        default=0,
        help="minimum number of speakers",
    )

    parser.add_argument(
        "--max_speakers",
        type=optional_int,
        default=0,
        help="maximum number of speakers",
    )

    parser.add_argument(
        "--args", nargs=argparse.REMAINDER
    )  # added to catch empty requests through shell script

    args = parser.parse_args(args)
    if not (args.input_dir or args.input):
        parser.error("At least one of -d or -i is required.")
    args = vars(args)

    args["input"] = aggregate_paths(args.get("input"), args.get("input_dir"))
    args.pop("input_dir")
    return args

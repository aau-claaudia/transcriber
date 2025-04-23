#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

from whispaau.cli_utils import file_duration


def test_file_duration():
    # Mock the input file
    file_path = Path("resources/end2end/input/shorts.m4a")

    # Call the function to get the duration
    duration = file_duration(file_path)

    # Verify that the duration is as expected
    assert duration == 4.992

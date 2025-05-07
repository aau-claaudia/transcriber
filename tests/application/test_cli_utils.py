#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest
from pathlib import Path

from whispaau.cli_utils import file_duration

class TestCliUtils(unittest.TestCase):
    def setUp(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.shorts_input = current_path / ".." / "resources" / "end2end" / "input" / "shorts.m4a"

    def test_file_duration(self):
        # Mock the input file
        file_path = Path(self.shorts_input)

        # Call the function to get the duration
        duration = file_duration(file_path)

        # Verify that the duration is as expected
        assert duration == 4.992

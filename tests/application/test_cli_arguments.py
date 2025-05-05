#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cli arguments """
import unittest
from pathlib import Path

from whispaau.cli_utils import parse_arguments
import pytest

class TestCliArguments(unittest.TestCase):

    # Create a fixture to simulate command-line arguments
    @pytest.fixture
    def mock_args(self):
        return [
            "-i",
            "resources/end2end/input/shorts.m4a",
            "-m",
            "tiny",
            "-o",
            "test_zip_out",
            "--no-mps",
            "--archive_password",
            "gg",
            "--job_name",
            "halløjsa d",
        ]


    def test_arguments_paths(self, mock_args):
        args = parse_arguments(mock_args)

        assert args["model"] == "tiny"
        assert args["no_mps"] == True
        assert args["no_cuda"] == False
        assert args["verbose"] == False
        assert args["input"] == {Path("resources/end2end/input/shorts.m4a").resolve()}
        assert args["output_dir"] == Path("test_zip_out")
        assert args["language"] == None
        assert args["logging"] == False
        assert args["threads"] == 0
        assert args["output_format"] == "all"
        assert args["prompt"] == []
        assert args["archive_password"] == "gg"
        assert args["job_name"] == "halløjsa d"

        assert "input_dir" not in args


    def test_parse_arguments_no_input(self):
        with pytest.raises(SystemExit):
            parse_arguments([])  # Simulate no input arguments, should raise an error

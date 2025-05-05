#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cli arguments """
import os
import unittest
from pathlib import Path

from whispaau.cli_utils import parse_arguments

class TestCliArguments(unittest.TestCase):

    def setUp(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        input_file_path = current_path / "somepath" / "shorts.m4a"
        # Initialize mock arguments for all tests
        self.mock_args = [
            "-i",
            input_file_path.__str__(),
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

    def test_arguments_paths(self):
        args = parse_arguments(self.mock_args)

        self.assertEqual(args["model"], "tiny")
        self.assertTrue(args["no_mps"])
        self.assertFalse(args["no_cuda"])
        self.assertFalse(args["verbose"])
        self.assertIsNotNone(args["input"])
        self.assertEqual(args["output_dir"], Path("test_zip_out"))
        self.assertIsNone(args["language"])
        self.assertFalse(args["logging"])
        self.assertEqual(args["threads"], 0)
        self.assertEqual(args["output_format"], "all")
        self.assertEqual(args["prompt"], [])
        self.assertEqual(args["archive_password"], "gg")
        self.assertEqual(args["job_name"], "halløjsa d")

        self.assertNotIn("input_dir", args)

    def test_parse_arguments_no_input(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])  # Simulate no input arguments, should raise an error


if __name__ == "__main__":
    unittest.main()
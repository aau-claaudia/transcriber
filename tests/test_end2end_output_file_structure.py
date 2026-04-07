import os
import unittest
from pathlib import Path

class TestDirectoryStructure(unittest.TestCase):
    def setUp(self):
        # Define the directory to check
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.directory_path = current_path / "output"

        # Define the expected files
        self.expected_files = [
            "DIALOGUE_small_en.aud",
            "DIALOGUE_small_en.csv",
            "DIALOGUE_small_en.docx",
            "DIALOGUE_small_en.dote.json",
            "DIALOGUE_small_en.json",
            "DIALOGUE_small_en_merged.csv",
            "DIALOGUE_small_en_merged.docx",
            "DIALOGUE_small_en_merged.dote.json",
            "DIALOGUE_small_en_merged.srt",
            "DIALOGUE_small_en_merged.vtt",
            "DIALOGUE_small_en.srt",
            "DIALOGUE_small_en.tsv",
            "DIALOGUE_small_en.txt",
            "DIALOGUE_small_en.vtt",
            "e2etest.zip",
            "shorts_small_da.aud",
            "shorts_small_da.csv",
            "shorts_small_da.docx",
            "shorts_small_da.dote.json",
            "shorts_small_da.json",
            "shorts_small_da_merged.csv",
            "shorts_small_da_merged.docx",
            "shorts_small_da_merged.dote.json",
            "shorts_small_da_merged.srt",
            "shorts_small_da_merged.vtt",
            "shorts_small_da.srt",
            "shorts_small_da.tsv",
            "shorts_small_da.txt",
            "shorts_small_da.vtt",
            "DIALOGUE_parakeet.csv",
            "DIALOGUE_parakeet.docx",
            "DIALOGUE_parakeet.dote.json",
            "DIALOGUE_parakeet.json",
            "DIALOGUE_parakeet_merged.csv",
            "DIALOGUE_parakeet_merged.docx",
            "DIALOGUE_parakeet_merged.dote.json",
            "DIALOGUE_parakeet_merged.srt",
            "DIALOGUE_parakeet_merged.vtt",
            "DIALOGUE_parakeet.srt",
            "DIALOGUE_parakeet.tsv",
            "DIALOGUE_parakeet.txt",
            "DIALOGUE_parakeet.vtt",
            "e2etest2.zip",
            "shorts_parakeet.aud",
            "shorts_parakeet.csv",
            "shorts_parakeet.docx",
            "shorts_parakeet.dote.json",
            "shorts_parakeet.json",
            "shorts_parakeet_merged.csv",
            "shorts_parakeet_merged.docx",
            "shorts_parakeet_merged.dote.json",
            "shorts_parakeet_merged.srt",
            "shorts_parakeet_merged.vtt",
            "shorts_parakeet.srt",
            "shorts_parakeet.tsv",
            "shorts_parakeet.txt",
            "shorts_parakeet.vtt",
            "transcribe.log"
        ]

    def test_directory_structure(self):
        print("Running end2end test - verifying folder structure for output files.")
        # Get the list of files in the directory
        actual_files = os.listdir(self.directory_path)

        # Check that all expected files are present
        for file in self.expected_files:
            self.assertIn(file, actual_files, f"Missing file: {file}")

        # Check that there are no unexpected files
        for actual in actual_files:
            self.assertIn(actual, self.expected_files, f"Unexpected file: {actual}")

import json
import os
from whispaau.writers import merge_speakers
from pathlib import Path
import unittest

class TestCsvWriter(unittest.TestCase):
    def setUp(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.transcription_with_same_speaker = current_path / "resources" / "transcription_with_same_speaker.json"
        self.expected_grouped_speakers = current_path / "resources" / "expected_grouped_speakers.json"

    def test_speaker_grouping(self):
        with open(self.transcription_with_same_speaker) as json_file:
            data = json.load(json_file)

        with open(self.expected_grouped_speakers) as json_file:
            expected_data = json.load(json_file)

        assert merge_speakers(data) == expected_data

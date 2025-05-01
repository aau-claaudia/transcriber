import json
import os
import unittest

from whispaau.writers import WriteCSV
from pathlib import Path

class TestCsvWriter(unittest.TestCase):
    def setUp(self):
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.test_file_path = current_path / "resources" / "test_output.csv"
        self.expected_file_path = current_path / "resources" / "expected_output_missing_speaker.csv"
        self.transcription_missing_speaker_path = current_path / "resources" / "shorts_missing_speaker_fist_line.json"
        self.transcription_with_bad_attribute_path = current_path / "resources" / "transcription_with_bad_attribute.json"

    def test_csv_writer(self):
        csv_writer = WriteCSV("")

        # read JSON test file
        with open(self.transcription_missing_speaker_path) as json_file:
            data = json.load(json_file)

        file = open(self.test_file_path, 'w')

        # write the CSV file
        csv_writer.write_result(data, file, {})

        expected_output_file = open(self.expected_file_path)

        # compare the generated file with the expected output
        with open(self.test_file_path, 'r') as file:
            output = file.read()
            expected_output = expected_output_file.read()
            assert output == expected_output, 'The generated output does not match the expected output.'

        # clean up after test, remove the generated test file
        os.remove(self.test_file_path)

        # verify bugfix for https://github.com/aau-claaudia/transcriber/issues/20
        csv_writer = WriteCSV("Test bugfix")

        # read JSON test file
        with open(self.transcription_with_bad_attribute_path) as json_file:
            data = json.load(json_file)

        file = open(self.test_file_path, 'w')
        # write the CSV file (should not give an error with bugfix)
        csv_writer.write_result(data, file, {})

        # clean up after test, remove the generated test file
        os.remove(self.test_file_path)
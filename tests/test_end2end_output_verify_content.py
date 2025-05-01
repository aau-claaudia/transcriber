import json

import unittest
import difflib

def read_file_as_string(file_path):
    """
    Reads the contents of a file and returns it as a string.
    :param file_path: Path to the file to be read.
    :return: Contents of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

class TranscriptionParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_and_concatenate(self):
        """
        Parses the JSON file and concatenates all 'text' values into a single string.
        """
        try:
            # Load the JSON file
            with open(self.file_path, 'r') as file:
                data = json.load(file)

            # Extract and concatenate all 'text' values
            concatenated_text = " ".join(item["text"] for item in data["lines"] if "text" in item)
            return concatenated_text
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading or parsing the file: {e}")
            return None
    def parse_first_speaker(self):
        """
        Parses the JSON file and returns the speaker from the first line
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data["lines"][0]["speakerDesignation"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading or parsing the file: {e}")
            return None


class TestTranscriptionOutput(unittest.TestCase):
    def setUp(self):
        parser = TranscriptionParser("output/DIALOGUE_small_en_merged.dote.json")
        result = parser.parse_and_concatenate()
        self.assertIsNotNone(result, "No text could be parsed from the generated output file!")
        self.generated_output_dialogue = result

        parser = TranscriptionParser("output/shorts_small_da_merged.dote.json")
        result = parser.parse_and_concatenate()
        self.assertIsNotNone(result, "No text could be parsed from the generated output file!")
        self.generated_output_shorts = result
        self.generated_first_speaker = parser.parse_first_speaker()

    def preprocess_text(self, text):
        # Preprocess text to ignore case, punctuation, and extra whitespace
        import re
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def calculate_similarity(self, text1, text2):
        # Calculate similarity using difflib
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def test_transcription_output(self):
        # Verify content for DIALOGUE_small_en_merged.dote.json
        # Load expected transcription output
        expected_output = read_file_as_string("resources/end2end/DIALOGUE_OUTPUT.txt")

        self.compare_fuzzy(expected_output, self.generated_output_dialogue, 0.88)

        # Verify content for shorts_small_da_merged.dote.json
        # Load expected transcription output
        expected_output = read_file_as_string("resources/end2end/SHORTS_OUTPUT.txt")

        self.compare_fuzzy(expected_output, self.generated_output_shorts, 0.88)

        # Verify the speaker has been created properly
        self.assertEqual(self.generated_first_speaker, "SPEAKER_00", "The speaker has not been created!")

    def compare_fuzzy(self, expected_output, generated_output, threshold):
        # Preprocess both expected and actual outputs
        generated = self.preprocess_text(generated_output)
        expected = self.preprocess_text(expected_output)
        # Calculate similarity
        similarity = self.calculate_similarity(generated, expected)
        print(f"Similariry: {similarity:.2%}")
        # Make assertion based on allowed similarity threshold
        self.assertGreaterEqual(similarity, threshold,
                                f"Transcription similarity ({similarity:.2%}) is below the threshold of {threshold:.2%}.")

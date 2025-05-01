import json
import os
from whispaau.writers import merge_speakers
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))
transcription_with_same_speaker = current_path / "resources" / "transcription_with_same_speaker.json"
expected_grouped_speakers = current_path / "resources" / "expected_grouped_speakers.json"

with open(transcription_with_same_speaker) as json_file:
    data = json.load(json_file)

with open(expected_grouped_speakers) as json_file:
    expected_data = json.load(json_file)

assert merge_speakers(data) == expected_data

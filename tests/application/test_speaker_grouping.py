import json
from whispaau.writers import merge_speakers

with open('../resources/transcription_with_same_speaker.json') as json_file:
    data = json.load(json_file)

with open('../resources/expected_grouped_speakers.json') as json_file:
    expected_data = json.load(json_file)

assert merge_speakers(data) == expected_data

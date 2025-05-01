import json
import os
from whispaau.writers import WriteCSV

csv_writer = WriteCSV("")

test_file_name: str = 'application/resources/test_output.csv'
expected_file_name: str = 'application/resources/expected_output_missing_speaker.csv'

# read JSON test file
with open('application/resources/shorts_missing_speaker_fist_line.json') as json_file:
    data = json.load(json_file)

file = open(test_file_name, 'w')

# write the CSV file
csv_writer.write_result(data, file, {})

expected_output_file = open(expected_file_name)

# compare the generated file with the expected output
with open(test_file_name, 'r') as file:
    output = file.read()
    expected_output = expected_output_file.read()
    assert output == expected_output, 'The generated output does not match the expected output.'

# clean up after test, remove the generated test file
os.remove(test_file_name)

# verify bugfix for https://github.com/aau-claaudia/transcriber/issues/20
csv_writer = WriteCSV("Test bugfix")

# read JSON test file
with open('application/resources/transcription_with_bad_attribute.json') as json_file:
    data = json.load(json_file)

file = open(test_file_name, 'w')
# write the CSV file (should not give an error with bugfix)
csv_writer.write_result(data, file, {})

# clean up after test, remove the generated test file
os.remove(test_file_name)
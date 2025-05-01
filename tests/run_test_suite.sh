#!/usr/bin/env bash

# script for running all tests for the transcriber application
# requires build and setup of virtual environment as prerequisite

# 1) run application tests
echo "Running test_cli_arguments.py"
python3 test_cli_arguments.py
echo "Running test_cli_utils.py"
python3 test_cli_utils.py
echo "Running test_utils.py"
python3 test_utils.py
echo "Running test_csv_writer.py"
python3 test_csv_writer.py
echo "Running test_logging.py"
python3 test_logging.py
echo "Running test_speaker_grouping.py"
python3 test_speaker_grouping.py


# 2) run the end2end test
# 2a) run the application with the e2e input files
echo "Starting end-to-end test"
echo "Running transcriber..."
python ../app.py -o ./output -d resources/end2end/input -m small --verbose --threads 4 --merge_speakers --no-cuda --no-mps --job_name e2etest

echo "Verifying output directory file structure"
echo "Running test_end2end_output_file_structure.py"
# 2b) verify file structure of output folder
python test_end2end_output_file_structure.py

echo "Verifying content of generated output files"
echo "Running test_end2end_output_verify_content.py"
# 2c) verify content of generated files
python test_end2end_output_verify_content.py
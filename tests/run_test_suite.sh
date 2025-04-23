#!/usr/bin/env bash

# script for running all tests for the transcriber application
# requires build and setup of virtual environment as prerequisite

# 1) run application tests
# TODO: comment in tests when finished with e2e
#echo "Running test_cli_arguments.py..."
#python3 test_cli_arguments.py
#echo "Running test_cli_utils.py..."
#python3 test_cli_utils.py
#echo "Running test_utils.py..."
#python3 test_utils.py
#echo "Running test_csv_writer.py..."
#python3 test_csv_writer.py
#echo "Running test_logging.py..."
#python3 test_logging.py
#echo "Running test_speaker_grouping.py..."
#python3 test_speaker_grouping.py


# 2) run the end2end test
# 2a) run the application with the e2e input files
echo "Starting end-to-end test..."
python3 ../app.py -o ./output -d ./ressources/end2end/input -m large-v3 --verbose --threads 4 --merge_speakers --no-cuda --no-mps --job_name e2etest

# 2b) verify file structure of output folder


# 2c) sample and verify content of generated files


# 2d) clean up after test - delete generated files
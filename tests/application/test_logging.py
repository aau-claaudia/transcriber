#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import logging
from pathlib import Path
from whispaau.logging import Logger


@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return tmp_path / "logs"


@pytest.fixture
def logger(temp_log_dir):
    return Logger("test_logger", temp_log_dir, verbose=True)


@pytest.fixture
def mock_ffmpeg_probe(mocker):
    # Mock the behavior of ffmpeg.probe
    mocker.patch("ffmpeg.probe", return_value={"format": {"duration": "10.5"}})


def test_logger_creation(logger, temp_log_dir):
    assert logger.name == "test_logger"
    assert logger.output_dir == temp_log_dir
    assert logger.verbose is True
    assert isinstance(logger.logger, logging.Logger)


def test_log_model_loading(logger, caplog):
    start_time = 0
    end_time = 1000
    logger.log_model_loading("ModelName", start_time, end_time)
    assert "Loading ModelName to" in caplog.text


def test_log_processing(logger, caplog):
    file_count = [0, 1, 2, 3, 4]
    logger.log_processing(file_count)
    assert "Processing #5.." in caplog.text


def test_log_file_start(logger, caplog, temp_log_dir):
    file_path = Path("resources/end2end/input/shorts.m4a")
    device = "CPU"
    logger.log_file_start(file_path, device)
    assert f"Starting {file_path.name}" in caplog.text
    assert f"on device: {device}" in caplog.text


def test_log_file_end(logger, caplog, temp_log_dir):
    file_path = temp_log_dir / "test_file.txt"
    start_time = 0
    end_time = 1000
    logger.log_file_end(file_path, start_time, end_time)
    assert f"Processed {file_path.name}" in caplog.text
    assert "it took" in caplog.text


def test_log_finished(logger, caplog):
    filename = "test.txt"
    logger.log_finished(filename)
    assert f"{filename} is finished" in caplog.text

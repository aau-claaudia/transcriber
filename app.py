#!/usr/bin/env python

import sys
import whisperx
import whisper
import gc
from pathlib import Path
from time import perf_counter_ns
from typing import List, Any
from whispaau.archive import archiving
from whispaau.cli_utils import parse_arguments
from whispaau.logging import Logger
from whispaau.utils import get_writer

# set the huggingface cache directories to directory with pre-downloaded model
# these settings must be made before importing torch
import os
PATH = '~/.cache/huggingface/predownloadedmodels/'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
import torch


def cli(args: dict[str, Any]) -> None:
    job_name = args.get("job_name")
    output_dir: Path = args.get("output_dir")
    output_dir.mkdir(exist_ok=True)
    verbose = args.get("verbose")
    log = Logger(name=job_name, output_dir=output_dir, verbose=verbose)

    # Setup CPU/GPU and model
    use_cuda = not args.get("no_cuda") and torch.cuda.is_available()
    use_mps = not args.get("no_mps") and torch.backends.mps.is_available()
    model_name = args.get("model")

    secret_password = args.get("archive_password", None)
    batch_size = 16 # this can be reduced if low on GPU memory

    if use_cuda:
        device = torch.device("cuda")
        compute_type = "float16"
    elif use_mps:
        device = torch.device("mps")
        compute_type = "float16"
    else:
        device = torch.device("cpu")
        compute_type = "int8"

    transcribe_arguments = {}
    if args.get("language", None):
        transcribe_arguments["language"] = args.get("language")

    if args.get("prompt", None):
        transcribe_arguments["prompt"] = args.get("prompt")

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)
        log.log_threads(threads)

    files = args.get("input")

    start_time = perf_counter_ns()
    # TODO: remove line below if we go ahead with original whisper for transcription
    # model = whisperx.load_model(model_name, device.type, compute_type=compute_type)
    model = whisper.load_model(model_name, device=device)

    log.log_model_loading(model_name, start_time, perf_counter_ns())

    output_format = args.pop("output_format")
    writer = get_writer(output_format, output_dir)

    log.log_processing(files)

    for file in files:
        options = {
            "highlight_words": None,
            "max_line_count": None,
            "max_line_width": None,
            "jobname": job_name,
            "filename": file,
        }
        process_file(
            log,
            file,
            output_dir,
            model_name,
            model,
            device,
            use_cuda,
            batch_size,
            writer,
            transcribe_arguments,
            options,
        )

    # Scan for generated files in output_dir:
    files_to_pack = [path for path in output_dir.glob("*") if path.is_file()]
    # Pack everything into a process_name.zip
    job_name_directory = Path(job_name)
    archiving(
        jobname=job_name_directory,
        output_file=output_dir / job_name_directory.with_suffix(".zip"),
        paths=files_to_pack,
        secret_password=secret_password,
    )


def process_file(
    log: Logger,
    file: Path,
    output_dir,
    model_name,
    model,
    device,
    use_cuda,
    batch_size,
    writer,
    trans_arguments,
    options: dict[str, Any],
) -> None:
    log.log_file_start(file, device)
    start_time = perf_counter_ns()

    audio = whisperx.load_audio(file.resolve().as_posix())

    # 1. Transcribe with original whisper (batched)
    # TODO: remove line below if we go ahead with original whisper for transcription (remove batch size and compute_type)
    # transcribed_result = model.transcribe(audio, batch_size=batch_size, **trans_arguments, )
    transcribed_result: dict[str, Any] = model.transcribe(
        file.resolve().as_posix(), **trans_arguments
    )
    language = transcribed_result["language"]

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=transcribed_result["language"], device=device)
    aligned_result = whisperx.align(transcribed_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # garbage collect memory
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=False, device=device)
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # TODO: add number of speakers as parameter to ucloud app?
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result: dict[str, Any] = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    result["language"] = language

    # transfer model specific attributes from whisper transcription to the diarized segments array
    transfer_model_attributes(transcribed_result["segments"], result["segments"])

    # write output files
    output_file = (
        output_dir / f"{file.stem}_{model_name}_{result.get('language', '--')}"
    )
    writer(
        result,
        output_file,
        options,
    )

    log.log_file_end(file, start_time, perf_counter_ns())


# This function copies the model specific attributes like "tokens" and "temperature" from the whisper output
# to the output generated by whisperx
def transfer_model_attributes(transcribed_result: List[dict[str, Any]], result: List[dict[str, Any]]):
    for i in range(len(transcribed_result)):
        result[i]["id"] = transcribed_result[i]["id"]
        result[i]["temperature"] = transcribed_result[i]["temperature"]
        result[i]["avg_logprob"] = transcribed_result[i]["avg_logprob"]
        result[i]["no_speech_prob"] = transcribed_result[i]["no_speech_prob"]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    cli_arguments = parse_arguments(None)
    cli(cli_arguments)

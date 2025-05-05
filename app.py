#!/usr/bin/env python

import sys
import whisperx.diarize
import whisper
import gc
from pathlib import Path
from time import perf_counter_ns
from typing import List, Any
from whispaau.archive import archiving
from whispaau.cli_utils import parse_arguments
from whispaau.logging import Logger
from whispaau.utils import get_writer
from whispaau.utils import WRITERS
from whispaau.utils import MERGED_SPEAKERS_WRITERS
from whispaau.utils import is_speaker_diarization_supported
from whispaau.writers import merge_speakers, reset_merge_speaker_data
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

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transcribe_arguments = {"fp16": False}
    if args.get("language", None):
        transcribe_arguments["language"] = args.get("language")

    if args.get("prompt", None):
        transcribe_arguments["prompt"] = args.get("prompt")

    # Create a dictionary for optional speaker parameters
    speaker_params = {}
    min_speakers = args.pop("min_speakers")
    if min_speakers > 0:
        speaker_params["min_speakers"] = min_speakers

    max_speakers = args.pop("max_speakers")
    if max_speakers > 0:
        speaker_params["max_speakers"] = max_speakers

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)
        log.log_threads(threads)

    files = args.get("input")

    start_time = perf_counter_ns()
    model = whisper.load_model(model_name, device=device)

    log.log_model_loading(model_name, start_time, perf_counter_ns())

    output_format = args.pop("output_format")
    writer = get_writer(output_format, output_dir, WRITERS)
    merge_writer = get_writer(output_format, output_dir, MERGED_SPEAKERS_WRITERS)

    log.log_processing(files)

    for file in files:
        options = {
            "highlight_words": None,
            "max_line_count": None,
            "max_line_width": None,
            "jobname": job_name,
            "filename": file
        }
        process_file(
            log,
            file,
            output_dir,
            model_name,
            model,
            device,
            use_cuda,
            writer,
            merge_writer,
            args.get("merge_speakers"),
            transcribe_arguments,
            options,
            speaker_params
        )
    if args.get("transcriber_gui"):
        # If running from the transcriber GUI the user can have multiple runs with different models
        # Get all available model names
        whisper_model_names = whisper.available_models()
        # Include files for all model names
        files_to_pack = [
        path
        for whisper_model_name in whisper_model_names
        for path in output_dir.glob(f"*_{whisper_model_name}_*")
        if path.is_file()
        ]
    else:
        files_to_pack = [path for path in output_dir.glob(f"*_{model_name}_*") if path.is_file()]

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
    writer,
    merge_writer,
    speaker_merge_enabled,
    trans_arguments,
    options: dict[str, Any],
    speaker_params,
) -> None:
    log.log_file_start(file, device)
    start_time = perf_counter_ns()

    # 1. Transcribe with original whisper
    transcribed_result: dict[str, Any] = model.transcribe(
        file.resolve().as_posix(), **trans_arguments
    )
    language = transcribed_result["language"]

    result: dict[str, Any] = {}
    # check if speaker diarization is supported for this language
    if is_speaker_diarization_supported(language):
        # 2. Align whisper output
        audio = whisperx.load_audio(file.resolve().as_posix())
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned_result = whisperx.align(transcribed_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        # garbage collect memory
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        del model_a

        # 3. Assign speaker labels
        # Initialize the diarization pipeline
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=False, device=device)
        diarize_segments = diarize_model(audio, **speaker_params)
        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        result["language"] = language

        # transfer model specific attributes from whisper transcription to the diarized segments array
        transfer_model_attributes(transcribed_result["segments"], result["segments"])
    else:
        # use whisper transcription output without speaker diarization
        result = transcribed_result
        speaker_merge_enabled = False
        print(f"Speaker diarization is disabled. There is no alignment model for this language.")

    # write output files
    output_file = (
        output_dir / f"{file.stem}_{model_name}_{result.get('language', '--')}"
    )
    writer(
        result,
        output_file,
        options,
    )
    # if speaker_merge_enabled then also write merged file formats
    if speaker_merge_enabled and (merge_writer is not None):
        output_file_merged_speakers = (
                output_dir / f"{file.stem}_{model_name}_{result.get('language', '--')}_merged"
        )
        merge_writer(
            merge_speakers(result),
            output_file_merged_speakers,
            options
        )
        # reset the merge speaker data
        reset_merge_speaker_data()

    log.log_file_end(file, start_time, perf_counter_ns())


# This function copies the model specific attributes like "avg_logprob" and "temperature" from the whisper output
# to the output generated by whisperx after speaker diarization
def transfer_model_attributes(transcribed_result: List[dict[str, Any]], result: List[dict[str, Any]]):
    for i in range(len(transcribed_result)):
        # safeguard against index error in case the length of the segment lists differ (can happen in rare cases if a wrong language has been detected)
        if i < len(result):
            result[i]["id"] = transcribed_result[i]["id"]
            result[i]["temperature"] = transcribed_result[i]["temperature"]
            result[i]["avg_logprob"] = transcribed_result[i]["avg_logprob"]
            result[i]["no_speech_prob"] = transcribed_result[i]["no_speech_prob"]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    cli_arguments = parse_arguments(None)
    cli(cli_arguments)

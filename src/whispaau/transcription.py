import os
import tempfile
import gc
from pathlib import Path
from typing import List, TypedDict, Any, Dict, Union, Tuple
from time import perf_counter_ns

import numpy as np
import torchaudio.transforms as T
import torch
import torchaudio
from whispaau.logging import Logger
from abc import ABC, abstractmethod
import whisper
import nemo.collections.asr as nemo_asr


class TranscriptionSegment(TypedDict):
    """Represents a single transcribed segment of audio."""
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

class TranscriptionResult(TypedDict):
    """Represents the full transcription output from the model."""
    text: str
    segments: List[TranscriptionSegment]
    language: str


class TranscriptionService(ABC):
    """Abstract base class for transcription services."""

    @abstractmethod
    def transcribe(self, file: Path, trans_arguments: Any) -> TranscriptionResult:
        """Transcribes an audio file and returns the result."""
        pass


# A module-level cache to store instantiated transcriber services.
# The key is a tuple (model_name, device).
_model_cache: Dict[Union[str, Tuple[str, str]], "TranscriptionService"] = {}


def _is_cuda_device(device: Any) -> bool:
    return str(device).startswith("cuda")


def _release_cuda_memory(device: Any) -> None:
    if _is_cuda_device(device) and torch.cuda.is_available():
        # Synchronize first so queued kernels complete before cache cleanup.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def transcribe(model_name: str, file: Path, trans_arguments: Any, device, log: Logger):
    """
    Instantiate and use the correct transcription class based on the model_name.
    This function uses a cache to store and reuse transcriber instances, ensuring
    that models are loaded into memory only once per model/device combination.
    """
    transcriber: TranscriptionService
    model_family = model_name.lower()

    # Reusing the same Parakeet CUDA model across files can leave stale GPU state,
    # especially after decoder failures. Use a fresh instance per file on CUDA.
    disable_cache_for_parakeet_cuda = "parakeet" in model_family and _is_cuda_device(device)

    if disable_cache_for_parakeet_cuda:
        transcriber = ParakeetTranscription(model_name, device, log)
        try:
            return transcriber.transcribe(file, trans_arguments)
        finally:
            # Drop model refs deterministically between files.
            if hasattr(transcriber, "model"):
                del transcriber.model
            gc.collect()
            _release_cuda_memory(device)

    cache_key: Union[str, Tuple[str, str]] = (model_name, str(device))
    if cache_key not in _model_cache:
        # This factory determines which transcription strategy to use.
        if model_family in whisper.available_models():
            _model_cache[cache_key] = WhisperTranscription(model_name, device, log)
        elif "parakeet" in model_family:
            _model_cache[cache_key] = ParakeetTranscription(model_name, device, log)
        else:
            log.get_logger().error(f"Unknown model family for model_name: {model_name}")
            raise ValueError(f"Unknown model family for model_name: {model_name}")

    transcriber = _model_cache[cache_key]
    return transcriber.transcribe(file, trans_arguments)


class WhisperTranscription(TranscriptionService):
    """ Transcription with openai whisper models """
    def __init__(self, model_name: str, device, log: Logger):
        self.log = log
        self.model_name = model_name
        start_time = perf_counter_ns()
        self.model = whisper.load_model(self.model_name, device=device)
        self.log.log_model_loading(self.model_name, start_time, perf_counter_ns())

    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
        transcription: dict[str, Any] = self.model.transcribe(
            file.resolve().as_posix(), **trans_arguments
        )
        return transcription


class ParakeetTranscription(TranscriptionService):
    """ Transcription with the NVIDIA model: nvidia/parakeet-tdt-0.6b-v3 """
    def __init__(self, model_name: str, device, log: Logger):
        self.log = log
        self.model_name = model_name
        self.device = device
        start_time = perf_counter_ns()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)
        self.log.log_model_loading(self.model_name, start_time, perf_counter_ns())
        self.sample_rate = 16000 # 16kHz
        self.chunk_length_s: int = 30 # 30 seconds
        self.buffering_threshold: int = 300 # 5 minutes
        self.long_form_configured = False
        self.model = self.model.to(device)

    def _configure_for_duration(self, duration: float) -> None:
        if duration > 30 and not self.long_form_configured:
            self.log.get_logger().info("Audio longer than 30 seconds, using long form transcription.")
            # Update self-attention model of fast-conformer encoder once.
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
            self.long_form_configured = True

    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
        audio = convert_to_16k_mono_audio(file.resolve().as_posix())
        duration = len(audio) / self.sample_rate
        self._configure_for_duration(duration)

        if duration > self.buffering_threshold:
            # if the audio is longer than 5 minutes transcribe in chunks to manage memory consumption
            return self.transcribe_buffered(audio)
        else:
            with torch.inference_mode():
                hypotheses = self.model.transcribe(
                    audio,
                    batch_size=1,
                    return_hypotheses=True
                )
            if hypotheses and len(hypotheses) > 0:
                segments, full_text = prepare_segments(hypotheses)

                # Prepare output data
                result = prepare_output(segments, full_text)
                return result
            else:
                return {}

    def transcribe_buffered(self, audio) -> dict[str, Any]:
        # Calculate chunk lengths in samples
        chunk_len_samples = self.chunk_length_s * self.sample_rate

        # Create a temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = []
            start = 0
            idx = 0
            while start < len(audio):
                end = start + chunk_len_samples
                chunk = audio[start:end]

                chunk_filepath = os.path.join(temp_dir, f"chunk_{idx}.wav")
                # Ensure chunk is a tensor and has the shape (channels, frames)
                chunk_tensor = torch.as_tensor(chunk)
                if chunk_tensor.ndim == 1:
                    chunk_tensor = chunk_tensor.unsqueeze(0)
                # save file
                torchaudio.save(chunk_filepath, chunk_tensor, self.sample_rate)

                chunks.append({
                    'audio_path': chunk_filepath,
                    'start_time': start / self.sample_rate,
                    'duration': len(chunk) / self.sample_rate
                })

                start += chunk_len_samples
                idx += 1
            self.log.get_logger().info(f"Audio was split into {len(chunks)} chunks.")

            # Transcribe all chunks
            self.log.get_logger().info(f"Transcribing chunks...")
            all_segments = []
            final_full_text: str = ""
            for i, chunk_info in enumerate(chunks):
                self.log.get_logger().info(f"Transcribing chunk {i + 1}/{len(chunks)} (duration: {chunk_info['duration']:.1f}s)...")
                # Transcribe chunk
                waveform, sr = torchaudio.load(str(chunk_info['audio_path']))
                with torch.inference_mode():
                    hypotheses = self.model.transcribe(
                        waveform.squeeze().to(torch.float32).numpy(),
                        batch_size=1,
                        return_hypotheses=True,
                    )
                if hypotheses and len(hypotheses) > 0:
                    segments, full_text = prepare_segments(hypotheses)
                    final_full_text += " " + full_text

                    for segment in segments:
                        segment['start'] += chunk_info['start_time']
                        segment['end'] += chunk_info['start_time']
                        all_segments.append(segment)
                self.log.get_logger().info(f"Chunk {i + 1} complete.")

            # Prepare output data
            result = prepare_output(all_segments, final_full_text)
            return result

def prepare_segments(hypotheses):
    # Transform the Hypothesis into a segment-based format
    segments = []
    full_text = []
    hypo = hypotheses[0]
    # Parakeet models typically have a frame shift of 0.08s (80ms)
    frame_shift = 0.08
    gap_threshold = 1.0  # Split segment if silence is > 1.0s

    words = hypo.text.split() if hypo.text else []
    num_words = len(words)

    if num_words == 0:
        return segments, ""

    # Some hypotheses may miss timestamp data entirely. Keep processing text with
    # default timings instead of crashing.
    raw_timestamps = getattr(hypo, "timestamp", [])
    timestamps = []
    for ts in raw_timestamps:
        try:
            timestamps.append(float(ts))
        except (TypeError, ValueError):
            continue

    num_ticks = len(timestamps)
    if num_ticks == 0:
        text = " ".join(words)
        segments.append({
            "text": text,
            "start": 0.0,
            "end": 0.0,
        })
        return segments, text

    current_words = []
    start_time = timestamps[0] * frame_shift

    for i in range(num_words):
        current_words.append(words[i])
        # Map word index to approximate timestamp index
        tick_idx = min(int((i / num_words) * num_ticks), num_ticks - 1)
        next_tick_idx = min(int(((i + 1) / num_words) * num_ticks), num_ticks - 1)

        pause_duration = (timestamps[next_tick_idx] - timestamps[tick_idx]) * frame_shift

        if (pause_duration > gap_threshold or i == num_words - 1) and current_words:
            text = " ".join(current_words)
            end_time = timestamps[tick_idx] * frame_shift
            if end_time < start_time:
                end_time = start_time
            segments.append({
                "text": text,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
            full_text.append(text)
            if i < num_words - 1:
                start_time = timestamps[next_tick_idx] * frame_shift
                current_words = []
    return segments, " ".join(full_text)

def prepare_output(all_segments, full_text):
    segments: List[TranscriptionSegment] = []
    for i, item in enumerate(all_segments, start=1):
        # create the TranscriptionSegment, missing fields are set to defaults
        seg: TranscriptionSegment = {
            "id": i,
            "seek": 0,
            "start": item["start"],
            "end": item["end"],
            "text": item["text"],
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0
        }
        segments.append(seg)
    result: TranscriptionResult = {
        "text": full_text,
        "segments": segments,
        "language": "en"
    }
    return result


"""Convert audio to mono 16kHz"""
def convert_to_16k_mono_audio(file: str, sample_rate: int = 16000) -> np.ndarray:
    waveform, sr = torchaudio.load(file)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if the sample rate doesn't match
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform.squeeze().to(torch.float32).numpy()
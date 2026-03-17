import os
import tempfile
from pathlib import Path
from typing import List, TypedDict, Any, Dict, Union, Tuple
from time import perf_counter_ns

import numpy as np
import librosa
import soundfile as sf

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


def transcribe(model_name: str, file: Path, trans_arguments: Any, device, log: Logger, duration: float):
    """
    Instantiate and use the correct transcription class based on the model_name.
    This function uses a cache to store and reuse transcriber instances, ensuring
    that models are loaded into memory only once per model/device combination.
    """
    transcriber: TranscriptionService
    model_family = model_name.lower()

    cache_key: Union[str, Tuple[str, str]] = (model_name, str(device))
    if cache_key not in _model_cache:
        # This factory determines which transcription strategy to use.
        if model_family in whisper.available_models():
            _model_cache[cache_key] = WhisperTranscription(model_name, device, log, duration)
        elif "canary" in model_family:
            _model_cache[cache_key] = CanaryTranscription(model_name, log, duration)
        elif "parakeet" in model_family:
            _model_cache[cache_key] = ParakeetTranscription(model_name, device, log, duration)
        else:
            log.get_logger().error(f"Unknown model family for model_name: {model_name}")
            raise ValueError(f"Unknown model family for model_name: {model_name}")

    transcriber = _model_cache[cache_key]
    return transcriber.transcribe(file, trans_arguments)


class WhisperTranscription(TranscriptionService):
    """ Transcription with openai whisper models """
    def __init__(self, model_name: str, device, log: Logger, duration: float):
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


class CanaryTranscription(TranscriptionService):
    """ Transcription with the NVIDIA canary model """
    def __init__(self, model_name: str, log: Logger, duration: float):
        pass

    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
        raise NotImplementedError("CanaryTranscription is not yet implemented.")


class ParakeetTranscription(TranscriptionService):
    """ Transcription with the NVIDIA parakeet model """
    def __init__(self, model_name: str, device, log: Logger, duration: float):
        self.log = log
        self.model_name = model_name
        self.duration = duration
        start_time = perf_counter_ns()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)
        self.log.log_model_loading(self.model_name, start_time, perf_counter_ns())
        self.sample_rate = 16000 # 16kHz
        self.chunk_length_s: int = 30
        self.overlap_s: int = 1
        self.batch_size: int = 16
        self.buffering_threshold: int = 300
        if duration > 30:
            log.get_logger().info("Audio longer than 30 seconds, using long form transcription.")
            # updating self-attention model of fast-conformer encoder
            # setting attention left and right context sizes to 256
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        self.model = self.model.to(device)

    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
        audio = convert_to_16k_mono_audio(self.sample_rate, file.resolve().as_posix())

        if self.duration > 60: #TODO: change to: self.duration > self.buffering_threshold:
            # if the audio is longer than 5 minutes transcribe in chunks to manage memory consumption
            return self.transcribe_buffered(audio)
        else:
            output = self.model.transcribe(audio, timestamps=True)
            # Extract text and timestamps
            result_data = output[0]
            text = result_data.text
            #word_timestamps = result_data.timestamp.get("word", [])
            segment_timestamps = result_data.timestamp.get("segment", [])
            # Prepare output data
            result = prepare_output(segment_timestamps, text)
            return result

    def transcribe_buffered(self, audio) -> dict[str, Any]:
        # Calculate chunk and overlap lengths in samples
        chunk_len_samples = self.chunk_length_s * self.sample_rate
        overlap_samples = 8000 # = self.overlap_s * self.sample_rate # TODO: experimenting with half a second
        step = chunk_len_samples - overlap_samples

        # Create a temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks = []
            start = 0
            idx = 0
            while start < len(audio):
                end = start + chunk_len_samples
                chunk = audio[start:end]

                chunk_filepath = os.path.join(temp_dir, f"chunk_{idx}.wav")
                sf.write(chunk_filepath, chunk, self.sample_rate)

                chunks.append({
                    'audio_path': chunk_filepath,
                    'start_time': start / self.sample_rate,
                    'duration': len(chunk) / self.sample_rate
                })

                start += step
                idx += 1
            self.log.get_logger().info(f"Audio was split into {len(chunks)} chunks.")

            # Transcribe all chunks
            self.log.get_logger().info(f"Transcribing chunks...")
            #all_words = []
            all_segments = []
            full_text: str = ""
            for i, chunk_info in enumerate(chunks):
                self.log.get_logger().info(f"Transcribing chunk {i + 1}/{len(chunks)} (duration: {chunk_info['duration']:.1f}s)...")
                # Transcribe chunk
                output = self.model.transcribe(
                    chunk_info['audio_path'],
                    batch_size=1,
                    timestamps=True,
                )

                result_data = output[0]
                chunk_text = result_data.text
                full_text.join(chunk_text)

                # Extract and adjust timestamps
                if hasattr(result_data, 'timestamp') and result_data.timestamp:
                    chunk_words = result_data.timestamp.get("word", [])
                    chunk_segments = result_data.timestamp.get("segment", [])

                    # Adjust timestamps by chunk start time
                    #for word in chunk_words:
                    #    word['start'] += chunk_info['start_time']
                    #    word['end'] += chunk_info['start_time']
                    #    all_words.append(word)

                    for segment in chunk_segments:
                        segment['start'] += chunk_info['start_time']
                        segment['end'] += chunk_info['start_time']
                        all_segments.append(segment)

                self.log.get_logger().info(f"Chunk {i + 1} complete: {len(chunk_text)} characters")

            # Prepare output data
            result = prepare_output(all_segments, full_text)
            return result


def prepare_output(all_segments, full_text):
    segments: List[TranscriptionSegment] = []
    for i, item in enumerate(all_segments, start=1):
        # create the TranscriptionSegment, missing fields are set to defaults
        seg: TranscriptionSegment = {
            "id": i,
            "seek": 0,
            "start": item["start"],
            "end": item["end"],
            "text": item["segment"],
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
def convert_to_16k_mono_audio(sample_rate, file: str) -> np.ndarray:
    audio_arr, sr = librosa.load(file, sr=None)
    if len(audio_arr.shape) > 1:
        audio_arr = librosa.to_mono(audio_arr)
    if sr != sample_rate:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=sample_rate)
    return audio_arr

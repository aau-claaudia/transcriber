from pathlib import Path
from typing import List, TypedDict, Any, Dict, Union, Tuple
from time import perf_counter_ns
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
    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
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
        start_time = perf_counter_ns()
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)
        self.log.log_model_loading(self.model_name, start_time, perf_counter_ns())
        if duration > 30:
            log.get_logger().info("Audio longer than 30 seconds, using long form transcription.")
            # updating self-attention model of fast-conformer encoder
            # setting attention left and right context sizes to 256
            self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        self.model = self.model.to(device)

    def transcribe(self, file: Path, trans_arguments: Any) -> dict[str, Any]:
        output = self.model.transcribe([file.resolve().as_posix()], timestamps=True)
        # TODO: create output format for further processing
        print(output)
        return {}

from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF

# Add additional supported alignment models
DEFAULT_ALIGN_MODELS_HF["fo"] = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-faroese-100h"
DEFAULT_ALIGN_MODELS_HF["sv"] = "KBLab/wav2vec2-large-voxrex-swedish"
DEFAULT_ALIGN_MODELS_HF["is"] = "m3hrdadfi/wav2vec2-large-xlsr-icelandic"
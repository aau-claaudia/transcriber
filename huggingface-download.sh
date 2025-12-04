#!/bin/bash

pip install -U "huggingface_hub[cli]"
hf auth login --token [hftoken]
hf download pyannote/speaker-diarization-3.1 --cache-dir ~/.cache/torch/pyannote/
hf download pyannote/segmentation-3.0 --cache-dir ~/.cache/torch/pyannote/

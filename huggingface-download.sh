#!/bin/bash

pip install -U "huggingface_hub[cli]"
huggingface-cli login --token [hftoken]
huggingface-cli download pyannote/speaker-diarization-3.1 --cache-dir ~/.cache/torch/pyannote/
huggingface-cli download pyannote/segmentation-3.0 --cache-dir ~/.cache/torch/pyannote/

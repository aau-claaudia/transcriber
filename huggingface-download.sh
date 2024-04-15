#!/bin/bash

pip install -U "huggingface_hub[cli]"
huggingface-cli login --token [insert your hugging face token here before running script]
huggingface-cli download pyannote/speaker-diarization-3.1 --cache-dir ~/.cache/huggingface/predownloadedmodels/
huggingface-cli download pyannote/segmentation-3.0 --cache-dir ~/.cache/huggingface/predownloadedmodels/


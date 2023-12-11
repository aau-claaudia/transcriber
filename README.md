# Transcriber

## Use of transcriber

```bash
$ pip install -r requirements.txt
```

or with docker

```bash
$ docker build --tag aautranscriber:latest .
```

## Usage

```bash
docker run -it --rm -v $(PWD):/app aautranscriber:latest -o ./output -i inputfile1.wav inputfile2.wav --threads 4
```

## usage:

python app.py

```
optional arguments:
-h, --help show this help message and exit
-m MODEL, --model MODEL what model is used?
--no-mps disables macOS GPU training
--no-cuda disables CUDA training
--verbose Print info to screen
-i INPUT [INPUT ...], --input INPUT [INPUT ...] Files for transcribring
-o OUTPUT_DIR, --output_dir OUTPUT_DIR Transcription output
-la LANGUAGE, --language LANGUAGE Language of audio, if not set let whisper guess
-l, --logging create log file
--threads THREADS number of threads used by torch for CPU inference
--prompt PROMPT [PROMPT ...]
```

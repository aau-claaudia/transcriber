import subprocess
from pathlib import Path
import tempfile
import os
from whispaau.logging import Logger

"""
The method ensures that the input file has a supported format, and otherwise attempts to convert the file to .mp3
Returns (False, ) if the input file format is not supported
"""
def pre_proces(input_file: Path, log: Logger) -> (bool, Path):
    # Get the file part name up until the first '.'
    file_name = input_file.name.split('.')[0]

    # if file format is directly supported, return True and the input file
    supported_formats = {'.mp3', '.mp4', '.m4a', '.wav', '.mpeg', '.mpg', '.wma'}
    if input_file.suffix.lower() in supported_formats:
        return True, input_file
    else:
        # try to convert file
        temp_dir = tempfile.mkdtemp(prefix="converted_mp3_")
        output_mp3_path = os.path.join(temp_dir, f"{input_file.stem}.mp3")
        return convert_to_mp3(input_file, output_mp3_path, log)

def convert_to_mp3(input_file, output_file, log: Logger) -> (bool, Path):
    """
    Converts an audio file to .mp3 using ffmpeg via subprocess.
    """
    command = [
        'ffmpeg',
        '-i', input_file,    # Input file
        '-vn',               # Disable video (useful if input is a video file)
        '-ar', '16000',      # Set audio sampling rate
        '-ac', '1',          # Set number of audio channels
        '-b:a', '128k',      # Set audio bitrate
        '-y',                # Accept overwrite
        output_file          # Output file
    ]

    try:
        # check=True will raise a CalledProcessError if the command fails
        # capture_output=True allows access to the error message if the command fails
        subprocess.run(command, check=True, capture_output=True, text=True)
        log.get_logger().info(f"Successfully converted input file to: {output_file}")
        log.get_logger().info(f"Running transcription on converted .mp3 file.")
        return True, Path(output_file)

    except subprocess.CalledProcessError as e:
        log.get_logger().info(f"Conversion failed for {input_file}.")
        log.get_logger().info(f"FFmpeg Error Output: {e.stderr}")
        return False, input_file

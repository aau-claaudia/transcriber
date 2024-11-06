import csv
import json
from datetime import datetime
from typing import TextIO

import docx
from whisperx.utils import ResultWriter, format_timestamp

# This dict holds the result of the transcription with grouped speakers
merged_speakers_result:  dict = {}
# This function groups the speaker lines with the same speaker into one
def merge_speakers(result: dict) -> dict:
    # only create the grouping one time
    if len(merged_speakers_result) == 0:
        # add the segments list and language
        merged_speakers_result["segments"] = []
        merged_speakers_result["language"] = result["language"]

        current_speaker: str = ""
        for line in result["segments"]:
            speaker, text = extract_speaker_and_text(line)
            if speaker != current_speaker:
                # create new dict and add to segments list
                new_segment = {
                    "speaker": speaker,
                    "text": text,
                    "start": line["start"],
                    "end": line["end"]
                }
                merged_speakers_result["segments"].append(new_segment)
                current_speaker = speaker
            else:
                # add the text to the previous dict in the segments list
                merged_speakers_result["segments"][-1]["text"] = (merged_speakers_result["segments"][-1]["text"].strip()
                                                                  + " "
                                                                  + text.strip())
                # push the end time forward
                merged_speakers_result["segments"][-1]["end"] = line["end"]

    return merged_speakers_result

def get_field_names(result: dict) -> list[str]:
    # make sure that the CSV header contains the 'speaker' header even though the first line has no speaker
    # if there is no speaker in the segment, try to take the keys from the next line
    # we don't want to hard code the speaker key into a fixed position in the header list
    for i in range(len(result)):
        if 'speaker' in result["segments"][i]:
            return list(result["segments"][i].keys())
    # if none of the lines had a speaker then just return the keys from the first line
    return list(result["segments"][0].keys())


class WriteCSV(ResultWriter):
    extension: str = "csv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        fieldnames: list[str] = get_field_names(result)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["segments"])


class WriteDOTE(ResultWriter):
    extension: str = "dote.json"

    @staticmethod
    def format_result(result: dict):
        interface = {"lines": []}
        for line in result["segments"]:
            speaker, text = extract_speaker_and_text(line)
            line_add = {                
                "startTime": format_timestamp(line["start"], True),
                "endTime": format_timestamp(line["end"], True),
                "speakerDesignation": speaker.strip(),
                "text": text.strip(),
            }
            interface["lines"].append(line_add)
        return interface

    def write_result(self, result: dict, file: TextIO, options: dict):
        transcription_data = self.format_result(result)
        json.dump(transcription_data, file)


class WriteDOCX(ResultWriter):
    extension: str = "docx"

    @staticmethod
    def format_time(
        start_time: int | float, end_time: int | float, max_time: int | float
    ) -> str:
        dt_start = datetime.utcfromtimestamp(start_time)
        dt_end = datetime.utcfromtimestamp(end_time)
        time_format = "%M:%S"
        if max_time >= 3600:
            time_format = "%H:%M:%S"
        return f"{dt_start.strftime(time_format)} - {dt_end.strftime(time_format)}"

    def write_result(self, result: dict, file: TextIO, options: dict):
        audio_filename = options.get("filename", "").stem
        document = docx.Document()
        document.add_heading(audio_filename, level=2)
        document.add_heading(options["jobname"], level=4)
        document.extended_properties.set_property("total_time", "   1")
        document.extended_properties.set_property(
            "application", "Whisper Transcription AAU extension"
        )
        document.core_properties.title = audio_filename
        document.core_properties.author = options.get("username", "")
        document.core_properties.subject = options.get("jobname", "")

        p = document.add_paragraph()
        max_time = result["segments"][-1]["end"]
        for line in result["segments"]:
            speaker, text = extract_speaker_and_text(line)
            time = p.add_run(self.format_time(line["start"], line["end"], max_time))
            time.italic = True
            p.add_run("\t")
            p.add_run(f'{speaker.strip()}\n')
            p.add_run("\t")
            p.add_run(f'{text.strip()}\n')
        document.save(file.name)


def extract_speaker_and_text(line):
    # if the algorithm was unable to determine the speaker, the line object will not have a speaker object
    if "speaker" in line:
        speaker = line["speaker"]
    else:
        speaker = "Undetermined speaker"
    # Also guard against missing texts if the algorithm was unable to transcribe the segment
    if "text" in line:
        text = line["text"]
    else:
        text = "Undetermined text"

    return speaker, text
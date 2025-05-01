from whispaau.utils import is_speaker_diarization_supported

assert is_speaker_diarization_supported("en") == True
assert is_speaker_diarization_supported("sv") == True
assert is_speaker_diarization_supported("fo") == True
assert is_speaker_diarization_supported("is") == True
assert is_speaker_diarization_supported("fr") == True
assert is_speaker_diarization_supported("de") == True
assert is_speaker_diarization_supported("it") == True
assert is_speaker_diarization_supported("da") == True
assert is_speaker_diarization_supported("no") == True

assert is_speaker_diarization_supported("haw") == False
assert is_speaker_diarization_supported("id") == False

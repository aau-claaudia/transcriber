import unittest

from whispaau.utils import is_speaker_diarization_supported

class TestUtils(unittest.TestCase):

    def test_speaker_diarization_support(self):
        assert is_speaker_diarization_supported("en") == True
        assert is_speaker_diarization_supported("sv") == True
        assert is_speaker_diarization_supported("fo") == True
        assert is_speaker_diarization_supported("is") == True
        assert is_speaker_diarization_supported("fr") == True
        assert is_speaker_diarization_supported("de") == True
        assert is_speaker_diarization_supported("it") == True
        assert is_speaker_diarization_supported("da") == True
        assert is_speaker_diarization_supported("no") == True
        assert is_speaker_diarization_supported("id") == True # Indonesian alignment model added in whisperx-3.8.6

        assert is_speaker_diarization_supported("haw") == False # No alignment model

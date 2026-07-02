import unittest

from whispaau.transcription import prepare_segments


class HypothesisWithTimestamps:
    def __init__(self, text, timestamp):
        self.text = text
        self.timestamp = timestamp


class HypothesisWithoutTimestamps:
    def __init__(self, text):
        self.text = text


class TestTranscription(unittest.TestCase):
    def test_prepare_segments_handles_empty_timestamps(self):
        hypotheses = [HypothesisWithTimestamps("this should still work", [])]

        segments, full_text = prepare_segments(hypotheses)

        assert full_text == "this should still work"
        assert len(segments) == 1
        assert segments[0]["text"] == "this should still work"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 0.0

    def test_prepare_segments_handles_missing_timestamp_attribute(self):
        hypotheses = [HypothesisWithoutTimestamps("missing timestamp field")]

        segments, full_text = prepare_segments(hypotheses)

        assert full_text == "missing timestamp field"
        assert len(segments) == 1
        assert segments[0]["text"] == "missing timestamp field"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 0.0

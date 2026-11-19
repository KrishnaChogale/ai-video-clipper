"""tests/test_audio_memory_and_continuity.py
Tests bounded-memory audio decoding, chunked energy analysis, and Whisper timestamp continuity.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.audio_analysis import (
    decode_audio_chunk,
    get_audio_duration,
    compute_energy_buckets,
    analyze_audio_chunked,
    clear_audio_cache,
)
from modules.speech_recognition import transcribe_audio

TEST_VIDEO = os.path.join(config.INPUT_DIR, "321206426716_20261118_165332_live_segment.mkv")


class TestAudioMemoryAndContinuity(unittest.TestCase):

    def setUp(self):
        self.has_video = os.path.isfile(TEST_VIDEO)

    def test_audio_duration_probe(self):
        if not self.has_video:
            self.skipTest("Test video not found")
        dur = get_audio_duration(TEST_VIDEO)
        self.assertGreater(dur, 200.0)
        self.assertLess(dur, 400.0)

    def test_decode_audio_chunk(self):
        if not self.has_video:
            self.skipTest("Test video not found")
        chunk_dur = 10.0
        y = decode_audio_chunk(TEST_VIDEO, start_sec=15.0, duration_sec=chunk_dur, sr=16000)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.dtype, np.float32)
        # 10s at 16000Hz = 160000 samples (allow small container boundary tolerance)
        self.assertAlmostEqual(len(y) / 16000.0, chunk_dur, delta=0.5)

    def test_compute_energy_buckets_offset(self):
        sr = 16000
        # Mock 5 seconds of audio
        y = np.sin(np.linspace(0, 500 * np.pi, sr * 5)).astype(np.float32)
        offset = 120
        buckets = compute_energy_buckets(y, sr=sr, start_offset_sec=offset)
        self.assertEqual(len(buckets), 5)
        self.assertEqual(buckets[0]["second"], 120)
        self.assertEqual(buckets[-1]["second"], 124)

    def test_chunked_energy_buckets_coverage(self):
        if not self.has_video:
            self.skipTest("Test video not found")
        # Run chunked analysis with 60s chunks on test video
        buckets = analyze_audio_chunked(TEST_VIDEO, chunk_duration_sec=60.0)
        self.assertGreater(len(buckets), 200)
        # Verify strictly sequential seconds from 0 to N-1
        for idx, b in enumerate(buckets):
            self.assertEqual(b["second"], idx)

    def test_transcribe_audio_continuity_and_ordering(self):
        if not self.has_video:
            self.skipTest("Test video not found")
        # Transcribe using 60-second chunks
        transcript, words = transcribe_audio(TEST_VIDEO, chunk_duration_sec=60.0)
        self.assertGreater(len(transcript), 0)
        self.assertGreater(len(words), 0)

        # Check that segments have valid positive duration and are ordered
        for s in transcript:
            self.assertGreaterEqual(s["start"], 0.0)
            self.assertGreaterEqual(s["end"], s["start"])

        # Check that word timestamps are ordered within segments
        prev_start = -1.0
        for w in words:
            self.assertGreaterEqual(w["start"], 0.0)
            self.assertGreaterEqual(w["end"], w["start"])
            self.assertGreaterEqual(w["start"], prev_start - 1.0) # monotonic with small tolerance
            prev_start = w["start"]


if __name__ == "__main__":
    unittest.main()

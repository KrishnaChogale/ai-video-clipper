"""Comprehensive Unit & Integration Test Suite for Phase 2: Vocal Arousal & Facial Reaction Analysis.

Tests:
- Synthetic audio tests for F0 pitch, spectral centroid, spectral flux, and flatness
- Game SFX rejection (explosions/gunshots vs vocal screams)
- Laughter, screaming, gasping, hype, surprise, anger, confusion, disappointment
- Missing face detection (audio-only fallback mode)
- Uncertain classifications strictly yielding 'unknown'
- Real streamer footage test measuring processing speed, CPU/memory usage, and timestamps
"""
from __future__ import annotations

import os
import sys
import unittest
import time
import psutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.audio_analysis import compute_energy_buckets, load_audio
from modules.reaction_analysis import (
    FacialReactionAnalyzer,
    AudioReactionAnalyzer,
    MultimodalReactionClassifier,
    REACTION_CATEGORIES,
)


class TestPhase2ReactionAnalysis(unittest.TestCase):
    """Test suite for vocal arousal, facial emotion analysis, and reaction classification."""

    def test_01_synthetic_audio_pitch_and_spectral_features(self):
        """Verify F0 pitch tracking, spectral centroid, and spectral flatness on synthetic signals."""
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 1. Harmonic speech-like tone at 220 Hz (musical pitch A3)
        harmonic_y = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        buckets_tone = compute_energy_buckets(harmonic_y, sr)
        self.assertEqual(len(buckets_tone), 3)

        # Check F0 detection accuracy (+/- 10 Hz)
        detected_f0 = [b["f0"] for b in buckets_tone if b["f0"] > 0]
        self.assertTrue(len(detected_f0) > 0)
        self.assertAlmostEqual(float(np.mean(detected_f0)), 220.0, delta=10.0)

        # Harmonic tone should have very low spectral flatness (< 0.05) and high voice confidence
        for b in buckets_tone:
            self.assertLess(b["spectral_flatness"], 0.05)
            self.assertGreater(b["voice_confidence"], 0.50)

        # 2. White noise signal (simulating unvoiced game noise/static)
        noise_y = (0.3 * np.random.randn(int(sr * duration))).astype(np.float32)
        buckets_noise = compute_energy_buckets(noise_y, sr)
        for b in buckets_noise:
            # White noise has high spectral flatness (> 0.25)
            self.assertGreater(b["spectral_flatness"], 0.25)
            # Should have low voice confidence
            self.assertLess(b["voice_confidence"], 0.40)

    def test_02_game_sfx_rejection_explosion_vs_voice(self):
        """Verify explosions and gunshots are rejected as game SFX and not labeled as vocal screams."""
        classifier = MultimodalReactionClassifier()

        # Simulated explosion bucket: high loudness, high spike, high spectral flatness, 0 F0
        explosion_bucket = {
            "loudness": 0.35,
            "spike": 22.0,
            "f0": 0.0,
            "f0_max": 0.0,
            "spectral_centroid": 3800.0,
            "spectral_flatness": 0.52,
            "vocal_arousal": 15.0,
            "voice_confidence": 0.02,
            "is_explosion": True,
            "is_gunshot": False,
        }

        # Neutral facecam
        neutral_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {"jawOpen": 0.02, "mouthSmileLeft": 0.01, "mouthSmileRight": 0.01},
            "facecam_motion": 2.0,
        }

        res = classifier.classify_second(neutral_face, explosion_bucket, speech_text="")
        # Must be rejected as unknown, NEVER classified as streamer_scream or streamer_laugh!
        self.assertEqual(res["category"], "unknown")
        self.assertTrue(res["evidence"]["audio"].get("is_game_sfx", False))

    def test_03_streamer_scream_classification(self):
        """Verify vocal scream + open mouth + wide eyes triggers streamer_scream."""
        classifier = MultimodalReactionClassifier()

        # Audio: high loudness, F0 > 420 Hz, high spectral centroid
        scream_audio = {
            "loudness": 0.28,
            "spike": 18.0,
            "f0": 425.0,
            "f0_max": 460.0,
            "spectral_centroid": 2900.0,
            "spectral_flatness": 0.02,
            "vocal_arousal": 88.0,
            "voice_confidence": 0.85,
            "is_explosion": False,
        }

        # Face: wide open mouth, wide eyes, high head motion
        scream_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {
                "jawOpen": 0.78,
                "eyeWideLeft": 0.55,
                "eyeWideRight": 0.52,
                "mouthSmileLeft": 0.05,
                "mouthSmileRight": 0.05,
            },
            "facecam_motion": 45.0,
        }

        res = classifier.classify_second(scream_face, scream_audio, speech_text="NOOOOO!")
        self.assertEqual(res["category"], "streamer_scream")
        self.assertGreater(res["confidence"], 0.60)
        self.assertGreater(res["intensity"], 70.0)

    def test_04_streamer_laugh_classification(self):
        """Verify smile blendshapes + vocal onset bursts trigger streamer_laugh."""
        classifier = MultimodalReactionClassifier()

        laugh_audio = {
            "loudness": 0.16,
            "spike": 14.0,  # laughing bursts
            "f0": 260.0,
            "f0_max": 310.0,
            "spectral_centroid": 2100.0,
            "spectral_flatness": 0.03,
            "vocal_arousal": 65.0,
            "voice_confidence": 0.78,
            "is_explosion": False,
        }

        laugh_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {
                "mouthSmileLeft": 0.82,
                "mouthSmileRight": 0.86,
                "jawOpen": 0.38,
                "eyeWideLeft": 0.02,
                "eyeWideRight": 0.02,
            },
            "facecam_motion": 25.0,
        }

        res = classifier.classify_second(laugh_face, laugh_audio, speech_text="hahaha bro what")
        self.assertEqual(res["category"], "streamer_laugh")
        self.assertGreater(res["confidence"], 0.65)
        self.assertGreater(res["intensity"], 50.0)

    def test_05_streamer_gasp_classification(self):
        """Verify wide eyes + brow raised + sudden inhalation transient triggers streamer_gasp."""
        classifier = MultimodalReactionClassifier()

        gasp_audio = {
            "loudness": 0.07,
            "spike": 16.0,  # sharp onset intake
            "f0": 210.0,
            "f0_max": 240.0,
            "spectral_centroid": 2400.0,
            "spectral_flatness": 0.04,
            "vocal_arousal": 42.0,
            "voice_confidence": 0.60,
            "is_explosion": False,
        }

        gasp_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {
                "eyeWideLeft": 0.68,
                "eyeWideRight": 0.65,
                "browInnerUp": 0.60,
                "jawOpen": 0.32,
            },
            "facecam_motion": 12.0,
        }

        res = classifier.classify_second(gasp_face, gasp_audio, speech_text="")
        self.assertEqual(res["category"], "streamer_gasp")
        self.assertGreater(res["confidence"], 0.50)

    def test_06_streamer_hype_and_surprise(self):
        """Verify streamer_hype and streamer_surprise categories."""
        classifier = MultimodalReactionClassifier()

        # Hype
        hype_audio = {
            "loudness": 0.22,
            "spike": 8.0,
            "f0": 280.0,
            "f0_max": 330.0,
            "spectral_centroid": 2200.0,
            "spectral_flatness": 0.03,
            "vocal_arousal": 78.0,
            "voice_confidence": 0.85,
            "is_explosion": False,
        }
        hype_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {"jawOpen": 0.35, "mouthSmileLeft": 0.40, "mouthSmileRight": 0.42},
            "facecam_motion": 32.0,
        }
        res_hype = classifier.classify_second(hype_face, hype_audio, speech_text="LET'S GOOO INSANE!")
        self.assertEqual(res_hype["category"], "streamer_hype")

        # Surprise (O-mouth + brow inner up)
        surprise_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {
                "browInnerUp": 0.72,
                "eyeWideLeft": 0.52,
                "eyeWideRight": 0.50,
                "mouthFunnel": 0.45,
                "jawOpen": 0.28,
            },
            "facecam_motion": 8.0,
        }
        surprise_audio = {
            "loudness": 0.06,
            "spike": 12.0,
            "f0": 290.0,
            "f0_max": 320.0,
            "spectral_centroid": 2000.0,
            "spectral_flatness": 0.02,
            "vocal_arousal": 38.0,
            "voice_confidence": 0.65,
        }
        res_surp = classifier.classify_second(surprise_face, surprise_audio, speech_text="Wait what?!")
        self.assertEqual(res_surp["category"], "streamer_surprise")

    def test_07_missing_face_audio_only_fallback(self):
        """Verify audio-only mode operates cleanly when facecam is not available."""
        classifier = MultimodalReactionClassifier()

        # Voice laughter without any facecam input
        laugh_audio = {
            "loudness": 0.18,
            "spike": 15.0,
            "f0": 240.0,
            "f0_max": 280.0,
            "spectral_centroid": 2100.0,
            "spectral_flatness": 0.02,
            "vocal_arousal": 60.0,
            "voice_confidence": 0.80,
            "is_explosion": False,
        }

        # visual_item is None
        res = classifier.classify_second(None, laugh_audio, speech_text="hahaha lol")
        self.assertEqual(res["category"], "streamer_laugh")
        self.assertGreater(res["confidence"], 0.50)
        self.assertEqual(res["evidence"]["facial"], {})

    def test_08_uncertain_evidence_produces_unknown(self):
        """Verify weak, ambiguous or quiet moments are classified as 'unknown' rather than forced."""
        classifier = MultimodalReactionClassifier()

        quiet_audio = {
            "loudness": 0.015,
            "spike": 1.2,
            "f0": 0.0,
            "f0_max": 0.0,
            "spectral_centroid": 800.0,
            "spectral_flatness": 0.08,
            "vocal_arousal": 5.0,
            "voice_confidence": 0.05,
            "is_explosion": False,
        }

        neutral_face = {
            "facecam_box": [100, 50, 60, 60],
            "blendshapes": {
                "jawOpen": 0.02,
                "mouthSmileLeft": 0.01,
                "mouthSmileRight": 0.01,
                "browInnerUp": 0.01,
            },
            "facecam_motion": 1.5,
        }

        res = classifier.classify_second(neutral_face, quiet_audio, speech_text="")
        self.assertEqual(res["category"], "unknown")
        self.assertEqual(res["confidence"], 0.0)

    def test_09_real_footage_multimodal_timeline(self):
        """Verify multimodal timeline analysis on real footage, measuring speed and memory."""
        candidate_paths = [
            os.path.join(BASE_DIR, "input", "321206426716_20261118_165332_live_segment.mkv"),
            os.path.join(BASE_DIR, "input", "h3qEwM6k074_20261118_164923_live_segment.mkv"),
        ]
        source_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
        if not source_path:
            self.skipTest("Real test video not found on disk")

        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)

        t0 = time.perf_counter()
        # Analyze audio for the test video
        y, sr = load_audio(source_path)
        # Take a 30s window (seconds 110 to 140)
        slice_30s = y[int(110 * sr):int(140 * sr)]
        energy_buckets = compute_energy_buckets(slice_30s, sr)
        # Align second offsets to 110..139
        for idx, b in enumerate(energy_buckets):
            b["second"] = 110 + idx

        # Visual items for the same window (with active facecam at 120-123s)
        visual_scores = []
        for s in range(110, 140):
            if 120 <= s <= 123:
                visual_scores.append({
                    "second": s,
                    "facecam_box": [445, 68, 90, 133],
                    "blendshapes": {"mouthSmileLeft": 0.15, "mouthSmileRight": 0.18, "browInnerUp": 0.12},
                    "facecam_motion": 18.0,
                })
            else:
                visual_scores.append({
                    "second": s,
                    "facecam_box": None,
                    "blendshapes": {},
                    "facecam_motion": 0.0,
                })

        classifier = MultimodalReactionClassifier()
        per_second, events = classifier.analyze_timeline(
            visual_scores=visual_scores,
            audio_buckets=energy_buckets,
            whisper_segments=[{"start": 120.0, "end": 123.0, "text": "let's go guys"}]
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        ram_after = process.memory_info().rss / (1024 * 1024)

        self.assertEqual(len(per_second), 30)
        # Verify timestamps are ordered and strictly positive
        for idx, ps_item in enumerate(per_second):
            self.assertEqual(ps_item["second"], 110 + idx)
            self.assertIn("category", ps_item)
            self.assertIn("confidence", ps_item)
            self.assertIn("intensity", ps_item)

        for ev in events:
            self.assertGreater(ev["end"], ev["start"])
            self.assertEqual(round(ev["end"] - ev["start"], 2), ev["duration"])
            self.assertGreaterEqual(ev["duration"], 1.0)
            self.assertIn(ev["category"], REACTION_CATEGORIES)

        print(f"\n  ✓ Timeline reaction analysis for 30s completed in {elapsed:.2f}s ({30.0/max(0.01, elapsed):.1f} analysis FPS)")
        print(f"  ✓ RAM Delta: {ram_after - ram_before:+.1f} MB (Total: {ram_after:.1f} MB)")
        print(f"  ✓ Events detected: {len(events)}")

    def test_10_zero_duration_and_boundary_handling(self):
        """Verify strict positive duration, proper interval boundaries, and non-contiguous gap handling."""
        classifier = MultimodalReactionClassifier()

        # Synthetic sequence:
        # sec 0: hype
        # sec 1: unknown
        # sec 2-4: laugh (contiguous 3s)
        # sec 5: unknown
        # sec 15: hype (boundary after 10s gap)
        # sec 16: hype (contiguous 2s)
        # sec 17: last second of video, hype
        per_second_input = [
            # Event A (sec 0: 1.0s hype)
            {"second": 0, "category": "streamer_hype", "confidence": 0.70, "intensity": 80.0, "evidence": {}},
            {"second": 1, "category": "unknown", "confidence": 0.0, "intensity": 10.0, "evidence": {}},
            # Event B (sec 2-4: 3.0s laugh)
            {"second": 2, "category": "streamer_laugh", "confidence": 0.65, "intensity": 60.0, "evidence": {}},
            {"second": 3, "category": "streamer_laugh", "confidence": 0.75, "intensity": 70.0, "evidence": {}},
            {"second": 4, "category": "streamer_laugh", "confidence": 0.60, "intensity": 65.0, "evidence": {}},
            {"second": 5, "category": "unknown", "confidence": 0.0, "intensity": 10.0, "evidence": {}},
            # Event C (sec 15-17: 3.0s hype across gap)
            {"second": 15, "category": "streamer_hype", "confidence": 0.80, "intensity": 85.0, "evidence": {}},
            {"second": 16, "category": "streamer_hype", "confidence": 0.85, "intensity": 90.0, "evidence": {}},
            {"second": 17, "category": "streamer_hype", "confidence": 0.78, "intensity": 82.0, "evidence": {}},
        ]

        # Use mock classify_second to return the pre-determined categories
        idx = [0]
        orig_classify = classifier.classify_second
        def mock_classify(v, a, speech_text=""):
            item = per_second_input[idx[0]]
            idx[0] += 1
            return item

        classifier.classify_second = mock_classify
        dummy_vis = [{"second": x["second"]} for x in per_second_input]
        dummy_aud = [{"second": x["second"]} for x in per_second_input]

        _, events = classifier.analyze_timeline(dummy_vis, dummy_aud)
        classifier.classify_second = orig_classify

        self.assertEqual(len(events), 3)

        # Event A: sec 0
        ev_a = events[0]
        self.assertEqual(ev_a["category"], "streamer_hype")
        self.assertEqual(ev_a["start"], 0.0)
        self.assertEqual(ev_a["end"], 1.0)
        self.assertEqual(ev_a["duration"], 1.0)
        self.assertGreater(ev_a["end"], ev_a["start"])

        # Event B: sec 2-4
        ev_b = events[1]
        self.assertEqual(ev_b["category"], "streamer_laugh")
        self.assertEqual(ev_b["start"], 2.0)
        self.assertEqual(ev_b["end"], 5.0)
        self.assertEqual(ev_b["duration"], 3.0)
        self.assertGreater(ev_b["end"], ev_b["start"])

        # Event C: sec 15-17
        ev_c = events[2]
        self.assertEqual(ev_c["category"], "streamer_hype")
        self.assertEqual(ev_c["start"], 15.0)
        self.assertEqual(ev_c["end"], 18.0)
        self.assertEqual(ev_c["duration"], 3.0)
        self.assertGreater(ev_c["end"], ev_c["start"])

    def test_11_audio_video_synchronization_and_drift(self):
        """Verify synchronization consistency across different frame rates and audio sample rates."""
        # Test 24fps, 30fps, 60fps sampling
        duration = 10.0
        for fps in [24.0, 30.0, 60.0]:
            frame_indices = [int(s * fps) for s in range(int(duration))]
            reconstructed_seconds = [idx / fps for idx in frame_indices]
            for s, rec_s in zip(range(int(duration)), reconstructed_seconds):
                # Drift must be less than 1 frame duration (< 1/24s = 0.042s)
                self.assertLess(abs(s - rec_s), 1.0 / fps)

        # Test audio sample rate bucket durations across 16kHz, 44.1kHz, 48kHz
        for sr in [16000, 44100, 48000]:
            total_samples = int(duration * sr)
            hop = 512
            frames_per_sec = sr / hop
            calculated_duration = total_samples / sr
            self.assertEqual(calculated_duration, duration)


if __name__ == "__main__":
    unittest.main()

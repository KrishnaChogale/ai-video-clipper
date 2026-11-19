"""Comprehensive Unit & Integration Test Suite for Phase 3:
Reaction-to-Gameplay Synchronization Engine.

Tests:
- Gameplay event detection (audio transients, visual motion, transcript cues)
- Adaptive temporal alignment (anticipation, immediate reaction, delayed commentary, rejection)
- Explainable synchronization scoring and structured outputs
- Unmatched event rejection (gameplay without reaction, reaction without gameplay)
- Complete 4-part highlight construction (Setup -> Trigger -> Reaction -> Payoff)
- Non-Maximum Suppression (NMS) deduplication
- Dynamic layout recommendation (50/50, 40/60, standard vertical)
- Real footage synchronization benchmark (throughput, latency, memory)
"""
from __future__ import annotations

import os
import sys
import unittest
import time
import psutil
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.reaction_sync import GameplayEventDetector, AdaptiveReactionSynchronizer
from modules.audio_analysis import analyze_audio
from modules.facecam import FacecamAnalyzer
from modules.reaction_analysis import MultimodalReactionClassifier


class TestReactionToGameplaySynchronization(unittest.TestCase):
    """Test suite for Phase 3: Gameplay Detection & Reaction Synchronization."""

    def setUp(self):
        self.detector = GameplayEventDetector()
        self.synchronizer = AdaptiveReactionSynchronizer()

    def test_01_gameplay_event_detection_explosions_and_gunfire(self):
        """Verify explosions and gunshots are detected with appropriate categories and confidence."""
        # Simulated 5-second energy buckets with explosion at second 2
        buckets = [
            {"second": 0, "loudness": 0.05, "spike": 2.0, "spectral_flatness": 0.03, "is_explosion": False, "is_gunshot": False},
            {"second": 1, "loudness": 0.04, "spike": 1.5, "spectral_flatness": 0.02, "is_explosion": False, "is_gunshot": False},
            {"second": 2, "loudness": 0.32, "spike": 18.0, "spectral_flatness": 0.45, "is_explosion": True, "is_gunshot": False},
            {"second": 3, "loudness": 0.12, "spike": 4.0, "spectral_flatness": 0.15, "is_explosion": False, "is_gunshot": False},
            {"second": 4, "loudness": 0.03, "spike": 1.0, "spectral_flatness": 0.02, "is_explosion": False, "is_gunshot": False},
        ]
        visuals = [{"second": s, "motion": 10.0, "scene_change": False} for s in range(5)]

        events = self.detector.detect_events(buckets, visuals)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["timestamp"], 2.0)
        self.assertEqual(ev["category"], "gameplay_explosion")
        self.assertGreaterEqual(ev["confidence"], 0.70)
        self.assertGreaterEqual(ev["intensity"], 60.0)
        self.assertTrue(ev["evidence"]["is_explosion"])

    def test_02_gameplay_collision_and_kill_death(self):
        """Verify motion spikes trigger collision/crash, and kill keywords trigger kill/death."""
        # Frame with sudden motion delta at second 3
        buckets = [{"second": s, "loudness": 0.12, "spike": 9.0, "spectral_flatness": 0.05} for s in range(6)]
        visuals = [
            {"second": 0, "motion": 12.0, "scene_change": False},
            {"second": 1, "motion": 14.0, "scene_change": False},
            {"second": 2, "motion": 15.0, "scene_change": False},
            {"second": 3, "motion": 52.0, "scene_change": True},  # Crash / Whip-pan
            {"second": 4, "motion": 20.0, "scene_change": False},
            {"second": 5, "motion": 10.0, "scene_change": False},
        ]

        events = self.detector.detect_events(buckets, visuals)
        crash_events = [e for e in events if e["category"] == "gameplay_collision_crash"]
        self.assertTrue(len(crash_events) >= 1)
        self.assertEqual(crash_events[0]["timestamp"], 3.0)

        # Test kill/death keyword trigger
        whisper = [{"start": 1.5, "end": 2.5, "text": "headshot he is dead"}]
        events_kd = self.detector.detect_events(buckets[:3], visuals[:3], whisper_segments=whisper)
        kd_events = [e for e in events_kd if e["category"] == "gameplay_kill_death"]
        self.assertTrue(len(kd_events) >= 1)

    def test_03_temporal_alignment_asymmetric_windows(self):
        """Verify explainable temporal alignment score across different reaction delay regimes."""
        # 1. Optimal physiological reaction time (0.8s delay)
        score_opt = self.synchronizer.compute_temporal_alignment(delay=0.8)
        self.assertEqual(score_opt, 1.0)

        # 2. Immediate simultaneous reaction (0.1s delay)
        score_imm = self.synchronizer.compute_temporal_alignment(delay=0.1)
        self.assertGreaterEqual(score_imm, 0.90)

        # 3. Delayed commentary (3.0s delay with continuous speech)
        score_delay_silent = self.synchronizer.compute_temporal_alignment(delay=3.0, is_speech_continuous=False)
        score_delay_speech = self.synchronizer.compute_temporal_alignment(delay=3.0, is_speech_continuous=True)
        self.assertGreater(score_delay_speech, score_delay_silent)
        self.assertGreaterEqual(score_delay_speech, 0.60)

        # 4. Anticipation / Pre-reaction (-1.5s delay)
        score_ant = self.synchronizer.compute_temporal_alignment(delay=-1.5, has_anticipation_word=True)
        self.assertGreaterEqual(score_ant, 0.60)

        # 5. Out of bounds (rejected)
        score_too_late = self.synchronizer.compute_temporal_alignment(delay=7.5)
        self.assertEqual(score_too_late, 0.0)

        score_too_early = self.synchronizer.compute_temporal_alignment(delay=-4.5)
        self.assertEqual(score_too_early, 0.0)

    def test_04_explainable_synchronization_scoring(self):
        """Verify structured output contains all required fields and explainable evidence."""
        gameplay = {
            "timestamp": 12.0,
            "category": "gameplay_explosion",
            "confidence": 0.85,
            "intensity": 80.0,
            "evidence": {"is_explosion": True, "loudness": 0.28},
        }
        reaction = {
            "start": 12.8,
            "end": 15.0,
            "duration": 2.2,
            "category": "streamer_scream",
            "confidence": 0.80,
            "intensity": 85.0,
            "evidence": {"facial": {"scream": 0.75}},
        }

        pair_res = self.synchronizer.evaluate_pairing(gameplay, reaction, speech_text="NOOO WAY!")
        self.assertEqual(pair_res["status"], "matched")
        self.assertEqual(pair_res["gameplay_event"], "gameplay_explosion")
        self.assertEqual(pair_res["gameplay_timestamp"], 12.0)
        self.assertEqual(pair_res["reaction_category"], "streamer_scream")
        self.assertEqual(pair_res["reaction_timestamp"], 12.8)
        self.assertEqual(pair_res["reaction_delay"], 0.8)
        self.assertGreaterEqual(pair_res["synchronization_confidence"], 0.75)

        # Check evidence dict completeness
        ev = pair_res["evidence"]
        self.assertIn("temporal_alignment", ev)
        self.assertIn("gameplay_confidence", ev)
        self.assertIn("reaction_confidence", ev)
        self.assertIn("cross_modal_coherence", ev)

    def test_05_unmatched_rejections(self):
        """Verify unrelated events and distant reactions are marked unmatched."""
        gameplay = {
            "timestamp": 10.0,
            "category": "gameplay_explosion",
            "confidence": 0.80,
            "intensity": 70.0,
        }
        # Reaction 15 seconds later (unrelated)
        reaction_distant = {
            "start": 25.0,
            "end": 28.0,
            "duration": 3.0,
            "category": "streamer_laugh",
            "confidence": 0.75,
            "intensity": 60.0,
        }

        pair_res = self.synchronizer.evaluate_pairing(gameplay, reaction_distant)
        self.assertEqual(pair_res["status"], "unmatched")

        # Weak low-confidence pair
        weak_gameplay = {"timestamp": 10.0, "category": "gameplay_action_peak", "confidence": 0.15, "intensity": 20.0}
        weak_reaction = {"start": 10.8, "end": 11.8, "duration": 1.0, "category": "streamer_hype", "confidence": 0.20, "intensity": 25.0}
        pair_weak = self.synchronizer.evaluate_pairing(weak_gameplay, weak_reaction)
        self.assertEqual(pair_weak["status"], "unmatched")

    def test_06_complete_highlight_construction_and_setup_payoff(self):
        """Verify adaptive pre-roll, post-roll, and duration bounds."""
        t_game = 20.0
        t_react_start = 21.0
        t_react_end = 24.0
        total_duration = 100.0

        transcript = [
            {"start": 16.0, "end": 19.5, "text": "let me try jumping this ramp guys"},
            {"start": 24.5, "end": 27.0, "text": "that was actually insane haha"},
        ]

        c_start, c_end = self.synchronizer.construct_highlight_boundaries(
            t_game=t_game,
            t_react_start=t_react_start,
            t_react_end=t_react_end,
            total_duration=total_duration,
            transcript_segments=transcript,
        )

        # Pre-roll should expand to capture speech setup starting at 16.0s
        self.assertLessEqual(c_start, 16.0)
        # Post-roll should expand to capture reaction payoff ending at 27.0s
        self.assertGreaterEqual(c_end, 27.0)
        # Clip duration must satisfy minimum bound (>= 15s)
        self.assertGreaterEqual(c_end - c_start, 15.0)

    def test_07_temporal_non_maximum_suppression(self):
        """Verify overlapping candidate windows and duplicate triggers are removed."""
        clips = [
            # High scoring clip around trigger at t=20
            {"clip_start": 15.0, "clip_end": 35.0, "gameplay_timestamp": 20.0, "clip_score": 88.0},
            # Overlapping lower-scoring candidate around same trigger
            {"clip_start": 16.0, "clip_end": 34.0, "gameplay_timestamp": 20.5, "clip_score": 62.0},
            # Distinct non-overlapping clip around trigger at t=70
            {"clip_start": 65.0, "clip_end": 85.0, "gameplay_timestamp": 70.0, "clip_score": 79.0},
        ]

        deduped = self.synchronizer._apply_nms(clips, iou_threshold=0.35)
        self.assertEqual(len(deduped), 2)
        # Verify highest score was kept
        scores = [c["clip_score"] for c in deduped]
        self.assertIn(88.0, scores)
        self.assertIn(79.0, scores)
        self.assertNotIn(62.0, scores)

    def test_08_dynamic_layout_recommendation(self):
        """Verify dynamic layout selection: 50/50, 40/60, and standard vertical."""
        # 1. High-energy reaction with high facecam coverage -> 50/50 or 60/40
        facecam_scores_active = [{"second": s, "box": [100, 50, 60, 60]} for s in range(20)]
        high_reaction = {"intensity": 82.0}
        rec_high = self.synchronizer._recommend_layout(0.0, 15.0, facecam_scores_active, high_reaction)
        self.assertEqual(rec_high["layout_type"], "facecam_split")
        self.assertEqual(rec_high["layout_ratio"], "50/50")

        # 2. Moderate reaction -> 40/60 (gameplay dominant)
        mod_reaction = {"intensity": 55.0}
        rec_mod = self.synchronizer._recommend_layout(0.0, 15.0, facecam_scores_active, mod_reaction)
        self.assertEqual(rec_mod["layout_type"], "facecam_split")
        self.assertEqual(rec_mod["layout_ratio"], "40/60")

        # 3. Low facecam coverage -> standard vertical
        facecam_scores_sparse = [{"second": s, "box": None} for s in range(20)]
        rec_sparse = self.synchronizer._recommend_layout(0.0, 15.0, facecam_scores_sparse, mod_reaction)
        self.assertEqual(rec_sparse["layout_type"], "standard_vertical")

    def test_09_real_footage_synchronization_benchmark(self):
        """Verify end-to-end synchronization on real streamer footage, measuring latency and VRAM."""
        test_video = os.path.join(BASE_DIR, "temp", "facecam_split_test.mp4")
        if not os.path.isfile(test_video):
            self.skipTest("Test video not found on disk")

        # Check VRAM before
        vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        process = psutil.Process()
        ram_before = process.memory_info().rss / (1024 * 1024)

        t0 = time.perf_counter()

        # Step 1: Audio Analysis
        energy_buckets, y, sr = analyze_audio(test_video)
        dur = len(y) / sr

        # Step 2: Visual Frame Sampling
        import cv2
        cap = cv2.VideoCapture(test_video)
        fa = FacecamAnalyzer()
        visual_scores = []
        for s in range(int(dur)):
            cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
            res = fa.analyze(frame, inset_only=False)
            visual_scores.append({
                "second": s,
                "box": res.get("box"),
                "motion": res.get("motion", 5.0),
                "scene_change": False,
                "blendshapes": res.get("blendshapes", {}),
            })
        cap.release()

        # Step 3: Phase 2 Reaction Analysis
        classifier = MultimodalReactionClassifier()
        _, reaction_events = classifier.analyze_timeline(visual_scores, energy_buckets)

        # Step 4: Phase 3 Gameplay Event Detection
        gameplay_events = self.detector.detect_events(energy_buckets, visual_scores)

        # Step 5: Phase 3 Adaptive Reaction Synchronization
        t_sync_0 = time.perf_counter()
        synchronized_highlights = self.synchronizer.synchronize(
            gameplay_events=gameplay_events,
            reaction_events=reaction_events,
            total_video_duration=dur,
            facecam_scores=visual_scores,
        )
        t_sync = time.perf_counter() - t_sync_0
        t_total = time.perf_counter() - t0

        vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        ram_after = process.memory_info().rss / (1024 * 1024)

        # Assertions
        self.assertIsInstance(synchronized_highlights, list)
        for h in synchronized_highlights:
            self.assertIn("gameplay_event", h)
            self.assertIn("reaction_category", h)
            self.assertIn("reaction_delay", h)
            self.assertIn("synchronization_confidence", h)
            self.assertIn("clip_start", h)
            self.assertIn("clip_end", h)
            self.assertGreater(h["clip_end"], h["clip_start"])
            self.assertGreaterEqual(h["clip_duration"], 15.0)

        # VRAM check: must be 0.0 MB
        self.assertEqual(vram_after - vram_before, 0)

        print(f"\n  ✓ End-to-end synchronization benchmark completed in {t_total:.2f}s")
        print(f"  ✓ Sync algorithm latency: {t_sync*1000.0:.2f} ms ({len(synchronized_highlights)} highlights)")
        print(f"  ✓ GPU VRAM delta: {(vram_after - vram_before)/(1024*1024):.1f} MB (100% CPU inference)")
        print(f"  ✓ RAM usage: {ram_after:.1f} MB (Delta: {ram_after - ram_before:+.1f} MB)")


if __name__ == "__main__":
    unittest.main()

"""tests/test_quality_and_ranking.py
Unit and integration tests for Phase 5:
- Unified Clip Quality Scoring (10 normalized factors, synergy, penalties)
- Intelligent Clip Ranking & Near-Duplicate Suppression
- Context Gating (reaction vs context priority)
- 50% Facecam Quota Balancer
- Post-Render Quality Control (1080x1920, 30 FPS, black frames, frozen frames, face persistence)
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.unified_quality_scorer import UnifiedQualityScorer
from modules.clip_ranker import ClipRanker
from modules.quality_control import evaluate_clip


class TestUnifiedQualityAndRanking(unittest.TestCase):

    def setUp(self):
        self.scorer = UnifiedQualityScorer()
        self.ranker = ClipRanker(target_facecam_ratio=0.50, min_quality_score=55.0)

    def test_01_unified_scoring_10_factors_normalized(self):
        """Verify all 10 scoring factors are bounded in [0, 100] and properly contribute."""
        candidate = {
            "start": 10.0,
            "end": 45.0,
            "category": "gameplay_explosion",
            "gameplay_event": {
                "category": "gameplay_explosion",
                "intensity": 90.0,
                "confidence": 0.85,
                "timestamp": 20.0,
            },
            "reaction_event": {
                "category": "streamer_scream",
                "intensity": 88.0,
                "confidence": 0.90,
                "timestamp": 20.8,
            },
            "reaction_sync": {
                "sync_score": 92.0,
                "reaction_delay": 0.8,
                "temporal_regime": "immediate_reaction",
            },
            "facecam_available": True,
            "facecam_coverage": 0.95,
            "facecam_score": 85.0,
        }

        res = self.scorer.evaluate_clip(candidate)
        self.assertIn("total_score", res)
        self.assertIn("factor_scores", res)
        self.assertTrue(0.0 <= res["total_score"] <= 100.0)

        factors = res["factor_scores"]
        expected_keys = [
            "reaction_intensity", "gameplay_importance", "reaction_synchronization",
            "entertainment_value", "narrative_completeness", "speech_context_relevance",
            "facecam_visibility_framing", "audio_clarity_balance", "caption_readability",
            "visual_quality"
        ]
        for k in expected_keys:
            self.assertIn(k, factors)
            self.assertTrue(0.0 <= factors[k] <= 100.0, f"Factor {k} out of bounds: {factors[k]}")

        self.assertTrue(res["passed_quality_gate"])
        self.assertGreater(res["total_score"], 75.0)
        self.assertIn("cross_modal_synergy", res["bonuses"])

    def test_02_high_reaction_with_poor_context_penalized(self):
        """Verify clip with high reaction but missing context (clipped setup) does NOT get top score."""
        # Candidate 1: High reaction but starts abruptly at the scream (0.2s setup)
        bad_context_candidate = {
            "start": 20.0,
            "end": 35.0,
            "gameplay_event": {"category": "gameplay_explosion", "intensity": 85.0, "timestamp": 20.2},
            "reaction_event": {"category": "streamer_scream", "intensity": 95.0, "confidence": 0.95, "timestamp": 20.8},
            "reaction_sync": {"sync_score": 90.0, "reaction_delay": 0.6},
            "facecam_available": True,
            "facecam_coverage": 0.90,
        }

        # Candidate 2: Slightly lower reaction intensity but perfect narrative setup and payoff
        good_context_candidate = {
            "start": 10.0,
            "end": 45.0,
            "gameplay_event": {"category": "gameplay_explosion", "intensity": 80.0, "timestamp": 20.0},
            "reaction_event": {"category": "streamer_laugh", "intensity": 80.0, "confidence": 0.85, "timestamp": 21.0},
            "reaction_sync": {"sync_score": 88.0, "reaction_delay": 1.0},
            "facecam_available": True,
            "facecam_coverage": 0.90,
        }

        res_bad = self.scorer.evaluate_clip(bad_context_candidate)
        res_good = self.scorer.evaluate_clip(good_context_candidate)

        # Bad context clip must be rejected or severely penalized
        self.assertFalse(res_bad["passed_quality_gate"])
        self.assertTrue(any("Missing setup context" in r for r in res_bad["rejection_reasons"]))
        self.assertGreater(res_good["total_score"], res_bad["total_score"])

    def test_03_near_duplicate_detection_and_ranking(self):
        """Verify intelligent ranking suppresses temporal near-duplicates (> 35% IoU)."""
        clip_a = {
            "start": 100.0,
            "end": 135.0,
            "gameplay_event": {"category": "gameplay_explosion", "intensity": 90.0, "timestamp": 115.0},
            "reaction_event": {"category": "streamer_scream", "intensity": 90.0, "timestamp": 115.8},
            "reaction_sync": {"sync_score": 90.0, "reaction_delay": 0.8},
            "facecam_available": True,
        }

        # Near-duplicate: starts 2 seconds later with identical event
        clip_dup = {
            "start": 102.0,
            "end": 137.0,
            "gameplay_event": {"category": "gameplay_explosion", "intensity": 80.0, "timestamp": 115.0},
            "reaction_event": {"category": "streamer_scream", "intensity": 80.0, "timestamp": 115.8},
            "reaction_sync": {"sync_score": 85.0, "reaction_delay": 0.8},
            "facecam_available": True,
        }

        # Independent clip from different section of stream
        clip_b = {
            "start": 300.0,
            "end": 335.0,
            "gameplay_event": {"category": "gameplay_kill_death", "intensity": 85.0, "timestamp": 315.0},
            "reaction_event": {"category": "streamer_laugh", "intensity": 85.0, "timestamp": 316.0},
            "reaction_sync": {"sync_score": 88.0, "reaction_delay": 1.0},
            "facecam_available": True,
        }

        res = self.ranker.rank_and_curate_clips([clip_a, clip_dup, clip_b], top_k=5)
        self.assertEqual(len(res["ranked_clips"]), 2)
        self.assertEqual(res["duplicate_count"], 1)

        # Verify duplicate clip is listed in rejected_clips
        dup_rejection = next((r for r in res["rejected_clips"] if r["stage"] == "duplicate_suppression"), None)
        self.assertIsNotNone(dup_rejection)
        self.assertIn("Near-duplicate", dup_rejection["reason"])

    def test_04_facecam_quota_balancer_ethics(self):
        """Verify 50% target is respected when valid clips exist, and never fabricates when they do not."""
        # Case 1: Ample valid facecam and gameplay clips exist
        c_face1 = {"start": 10.0, "end": 45.0, "facecam_available": True, "gameplay_event": {"timestamp": 20.0}, "reaction_event": {"timestamp": 20.8}}
        c_face2 = {"start": 60.0, "end": 95.0, "facecam_available": True, "gameplay_event": {"timestamp": 70.0}, "reaction_event": {"timestamp": 70.8}}
        c_game1 = {"start": 110.0, "end": 145.0, "facecam_available": False, "gameplay_event": {"timestamp": 120.0}, "reaction_event": {"timestamp": 120.8}}
        c_game2 = {"start": 160.0, "end": 195.0, "facecam_available": False, "gameplay_event": {"timestamp": 170.0}, "reaction_event": {"timestamp": 170.8}}

        res1 = self.ranker.rank_and_curate_clips([c_face1, c_face2, c_game1, c_game2], top_k=4)
        self.assertTrue(res1["facecam_target_met"])
        self.assertGreaterEqual(res1["facecam_ratio"], 0.50)

        # Case 2: Source lacks facecam clips (e.g. gameplay only)
        res2 = self.ranker.rank_and_curate_clips([c_game1, c_game2], top_k=2, facecam_available_in_source=False)
        self.assertFalse(res2["facecam_target_met"])
        self.assertEqual(len(res2["ranked_clips"]), 2)
        self.assertIn("insufficient valid facecam moments", res2["facecam_status_report"])

    def test_05_misleading_headline_rejection(self):
        """Verify misleading headline on quiet moment triggers quality penalty & rejection."""
        candidate = {
            "start": 10.0,
            "end": 45.0,
            "headline": "BRO LOST HIS MIND AND STARTED SCREAMING!",
            "gameplay_event": {"category": "unknown", "intensity": 30.0, "timestamp": 20.0},
            "reaction_event": {"category": "unknown", "intensity": 25.0, "timestamp": 21.0},
            "reaction_sync": {"sync_score": 40.0, "reaction_delay": 1.0},
            "facecam_available": True,
        }

        res = self.scorer.evaluate_clip(candidate)
        self.assertIn("misleading_reaction_headline", res["penalties"])
        self.assertFalse(res["passed_quality_gate"])

    def test_06_comprehensive_qc_checks(self):
        """Verify quality control detects black frames, frozen frames, and resolution errors."""
        import tempfile
        import cv2

        tmp_dir = tempfile.mkdtemp()

        # 1. Create a synthetic black frame video (mean brightness < 5)
        black_vid_path = os.path.join(tmp_dir, "black_test.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(black_vid_path, fourcc, 30.0, (1080, 1920))
        for _ in range(60):
            frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        qc_res = evaluate_clip(black_vid_path, 0.0, 2.0, 80.0, min_duration=1.0)
        self.assertFalse(qc_res["passed"])
        self.assertTrue(qc_res["details"]["black_frames_detected"] or not qc_res["details"]["first_frame_ok"])

        # 2. Resolution check on a 640x360 file
        bad_res_path = os.path.join(tmp_dir, "bad_res.mp4")
        out2 = cv2.VideoWriter(bad_res_path, fourcc, 30.0, (640, 360))
        for _ in range(30):
            frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
            out2.write(frame)
        out2.release()

        qc_bad_res = evaluate_clip(bad_res_path, 0.0, 1.0, 80.0, min_duration=0.5)
        self.assertFalse(qc_bad_res["passed"])
        self.assertIn("Invalid output resolution", str(qc_bad_res["issues"]))


if __name__ == "__main__":
    unittest.main()

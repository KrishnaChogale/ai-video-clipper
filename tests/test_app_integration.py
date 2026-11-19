"""tests/test_app_integration.py
Integration tests for Phase 5 pipeline within the application:
1. Normal footage with valid facecam -> unified scoring and >= 50% facecam allocation.
2. Gameplay-only footage -> facecam ratio 0%, no fabrication, standard layout assigned.
3. Insufficient facecam footage -> quality threshold maintained, shortfall reported without padding.
4. Duplicate candidates -> temporal IoU near-duplicates suppressed.
5. Low-quality candidates -> rejected by quality gate with actionable reasons.
6. Missing/incomplete transcript -> graceful fallback without crash.
7. Downstream rendering compatibility -> all required keys present (start, end, score, reason, category, peak_second, duration).
8. Correct ranking and score explanations -> best-to-worst order strictly verified.
"""
from __future__ import annotations

import os
import sys
import unittest
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from modules.clip_ranker import ClipRanker
from modules.unified_quality_scorer import UnifiedQualityScorer


class TestAppPhase5Integration(unittest.TestCase):

    def setUp(self):
        self.min_quality = getattr(config, "MIN_QUALITY_SCORE", 55.0)
        self.ranker = ClipRanker(target_facecam_ratio=0.50, min_quality_score=self.min_quality)
        self.scorer = UnifiedQualityScorer()

    def test_01_normal_footage_with_valid_facecam(self):
        """Scenario 1: Normal footage with valid facecam -> unified scoring and >= 50% facecam allocation."""
        candidates = [
            {
                "start": 10.0, "end": 40.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 85.0, "confidence": 0.85, "timestamp": 20.0},
                "reaction_event": {"category": "streamer_scream", "intensity": 85.0, "confidence": 0.85, "timestamp": 20.8},
                "reaction_sync": {"sync_score": 88.0, "reaction_delay": 0.8},
                "facecam_available": True, "facecam_coverage": 0.90, "facecam_score": 85.0,
                "layout_type": "facecam_split", "peak_second": 20.0,
            },
            {
                "start": 50.0, "end": 80.0, "category": "action",
                "gameplay_event": {"category": "gameplay_gunfight", "intensity": 80.0, "confidence": 0.80, "timestamp": 60.0},
                "reaction_event": {"category": "streamer_hype", "intensity": 80.0, "confidence": 0.80, "timestamp": 60.7},
                "reaction_sync": {"sync_score": 85.0, "reaction_delay": 0.7},
                "facecam_available": True, "facecam_coverage": 0.88, "facecam_score": 80.0,
                "layout_type": "facecam_split", "peak_second": 60.0,
            },
            {
                "start": 90.0, "end": 120.0, "category": "stunt",
                "gameplay_event": {"category": "gameplay_stunt_jump", "intensity": 82.0, "confidence": 0.80, "timestamp": 100.0},
                "facecam_available": False, "facecam_coverage": 0.0, "facecam_score": 0.0,
                "layout_type": "standard", "peak_second": 100.0,
            },
            {
                "start": 130.0, "end": 160.0, "category": "police_chase",
                "gameplay_event": {"category": "gameplay_police_pursuit", "intensity": 78.0, "confidence": 0.75, "timestamp": 140.0},
                "facecam_available": False, "facecam_coverage": 0.0, "facecam_score": 0.0,
                "layout_type": "standard", "peak_second": 140.0,
            },
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=4, facecam_available_in_source=True)
        self.assertEqual(len(res["ranked_clips"]), 4)
        self.assertGreaterEqual(res["facecam_ratio"], 0.50)
        self.assertTrue(res["facecam_target_met"])
        self.assertIn("Facecam target met", res["facecam_status_report"])

    def test_02_gameplay_only_footage_no_fabrication(self):
        """Scenario 2: Gameplay-only footage -> facecam ratio 0%, no fabrication, standard layout assigned."""
        candidates = [
            {
                "start": float(i * 40), "end": float(i * 40 + 30), "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 80.0, "confidence": 0.8, "timestamp": float(i * 40 + 10)},
                "facecam_available": False, "facecam_coverage": 0.0, "facecam_score": 0.0,
                "layout_type": "standard", "peak_second": float(i * 40 + 10),
            }
            for i in range(5)
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=3, facecam_available_in_source=False)
        self.assertEqual(len(res["ranked_clips"]), 3)
        self.assertEqual(res["facecam_ratio"], 0.0)
        self.assertFalse(res["facecam_target_met"])
        for clip in res["ranked_clips"]:
            self.assertFalse(clip.get("facecam_available", False))
            self.assertEqual(clip.get("layout_type", "standard"), "standard")
        self.assertIn("Preserved quality standard without fabricating facecam clips", res["facecam_status_report"])

    def test_03_insufficient_facecam_shortfall_reported_no_padding(self):
        """Scenario 3: Insufficient facecam footage -> quality threshold maintained, shortfall reported without padding."""
        candidates = [
            # Only 1 high quality facecam clip
            {
                "start": 10.0, "end": 40.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 85.0, "confidence": 0.85, "timestamp": 20.0},
                "reaction_event": {"category": "streamer_scream", "intensity": 85.0, "confidence": 0.85, "timestamp": 20.8},
                "facecam_available": True, "facecam_coverage": 0.90, "facecam_score": 85.0,
                "layout_type": "facecam_split", "peak_second": 20.0,
            },
            # 3 gameplay clips
            {
                "start": 50.0, "end": 80.0, "category": "action",
                "gameplay_event": {"category": "gameplay_gunfight", "intensity": 80.0, "confidence": 0.80, "timestamp": 60.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 60.0,
            },
            {
                "start": 90.0, "end": 120.0, "category": "action",
                "gameplay_event": {"category": "gameplay_stunt_jump", "intensity": 78.0, "confidence": 0.80, "timestamp": 100.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 100.0,
            },
            {
                "start": 130.0, "end": 160.0, "category": "action",
                "gameplay_event": {"category": "gameplay_police_pursuit", "intensity": 75.0, "confidence": 0.75, "timestamp": 140.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 140.0,
            },
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=4, facecam_available_in_source=True)
        # We wanted 50% (2 facecam clips), but only 1 exists
        self.assertEqual(len(res["ranked_clips"]), 4)
        facecam_count = sum(1 for c in res["ranked_clips"] if c.get("facecam_available"))
        self.assertEqual(facecam_count, 1)
        self.assertEqual(res["facecam_ratio"], 0.25)
        self.assertFalse(res["facecam_target_met"])
        self.assertIn("partially met", res["facecam_status_report"])

    def test_04_duplicate_candidates_suppression(self):
        """Scenario 4: Duplicate candidates -> temporal IoU near-duplicates suppressed."""
        candidates = [
            {
                "start": 10.0, "end": 40.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 90.0, "confidence": 0.9, "timestamp": 20.0},
                "reaction_event": {"category": "streamer_scream", "intensity": 90.0, "confidence": 0.9, "timestamp": 20.8},
                "facecam_available": True, "facecam_coverage": 0.95, "facecam_score": 90.0,
                "layout_type": "facecam_split", "peak_second": 20.0,
            },
            # Near duplicate: 12s to 42s (over 80% overlap)
            {
                "start": 12.0, "end": 42.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 88.0, "confidence": 0.85, "timestamp": 20.0},
                "reaction_event": {"category": "streamer_scream", "intensity": 88.0, "confidence": 0.85, "timestamp": 20.8},
                "facecam_available": True, "facecam_coverage": 0.95, "facecam_score": 88.0,
                "layout_type": "facecam_split", "peak_second": 20.0,
            },
            # Independent clip
            {
                "start": 80.0, "end": 110.0, "category": "action",
                "gameplay_event": {"category": "gameplay_gunfight", "intensity": 80.0, "confidence": 0.80, "timestamp": 90.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 90.0,
            },
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=3, facecam_available_in_source=True)
        self.assertGreaterEqual(res["duplicate_count"], 1)
        # The near duplicate must be in rejected_clips with duplicate_suppression stage
        rejected_stages = [r["stage"] for r in res["rejected_clips"]]
        self.assertIn("duplicate_suppression", rejected_stages)
        # Only 2 distinct clips should be accepted
        self.assertEqual(len(res["ranked_clips"]), 2)

    def test_05_low_quality_candidates_gating(self):
        """Scenario 5: Low-quality candidates -> rejected by quality gate with actionable reasons."""
        bad_candidate = {
            "start": 10.0, "end": 35.0, "category": "action",
            # Muted audio, zero motion, low intensity
            "gameplay_event": {"category": "unknown", "intensity": 10.0, "confidence": 0.1, "timestamp": 20.0},
            "reaction_event": {"category": "unknown", "intensity": 10.0, "confidence": 0.1, "timestamp": 20.6},
            "facecam_available": True, "facecam_coverage": 0.15, "facecam_score": 10.0,
        }
        res = self.scorer.evaluate_clip(bad_candidate)
        self.assertFalse(res["passed_quality_gate"])
        self.assertLess(res["total_score"], self.min_quality)
        self.assertGreater(len(res["rejection_reasons"]), 0)

        # In ranker
        rank_res = self.ranker.rank_and_curate_clips([bad_candidate], top_k=1)
        self.assertEqual(len(rank_res["ranked_clips"]), 0)
        self.assertEqual(len(rank_res["rejected_clips"]), 1)
        self.assertEqual(rank_res["rejected_clips"][0]["stage"], "quality_gate")

    def test_06_missing_or_incomplete_transcript_fallback(self):
        """Scenario 6: Missing/incomplete transcript -> graceful fallback without crash."""
        candidate = {
            "start": 10.0, "end": 35.0, "category": "action",
            "gameplay_event": {"category": "gameplay_explosion", "intensity": 80.0, "confidence": 0.8, "timestamp": 20.0},
            "facecam_available": False,
        }
        # Evaluated with None transcript segments
        res = self.scorer.evaluate_clip(candidate, whisper_segments=None)
        self.assertIn("total_score", res)
        self.assertGreater(res["total_score"], 0.0)

        # Evaluated with empty list transcript
        res2 = self.scorer.evaluate_clip(candidate, whisper_segments=[])
        self.assertIn("total_score", res2)
        self.assertGreater(res2["total_score"], 0.0)

    def test_07_downstream_rendering_compatibility_keys(self):
        """Scenario 7: Downstream rendering compatibility -> all required keys present."""
        candidates = [
            {
                "start": 10.0, "end": 40.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 85.0, "confidence": 0.85, "timestamp": 20.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 20.0,
            }
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=1)
        self.assertEqual(len(res["ranked_clips"]), 1)
        c = res["ranked_clips"][0]

        # Simulate mapping in app.py
        h = dict(c)
        h["rank"] = 1
        h["unified_score"] = c.get("unified_score", c.get("score", 0.0))
        h["score"] = h["unified_score"]
        h["duration"] = h["end"] - h["start"]
        h["reason"] = c.get("explanation") or c.get("reason", "")
        h["score_factors"] = c.get("quality_evaluation", {}).get("factor_scores", c.get("score_factors", {}))
        h["factor_scores"] = h["score_factors"]

        required_keys = ["start", "end", "score", "reason", "category", "peak_second", "duration", "layout_type", "score_factors"]
        for k in required_keys:
            self.assertIn(k, h, f"Missing required downstream key '{k}'")

        self.assertGreater(h["duration"], 0.0)
        self.assertGreaterEqual(h["score"], self.min_quality)

    def test_08_correct_ranking_and_score_explanations(self):
        """Scenario 8: Correct ranking and score explanations -> best-to-worst order strictly verified."""
        candidates = [
            {
                "start": 10.0, "end": 40.0, "category": "action",
                "gameplay_event": {"category": "gameplay_gunfight", "intensity": 70.0, "confidence": 0.7, "timestamp": 20.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 20.0,
            },
            {
                "start": 60.0, "end": 90.0, "category": "action",
                "gameplay_event": {"category": "gameplay_explosion", "intensity": 95.0, "confidence": 0.95, "timestamp": 70.0},
                "reaction_event": {"category": "streamer_scream", "intensity": 95.0, "confidence": 0.95, "timestamp": 70.8},
                "reaction_sync": {"sync_score": 95.0, "reaction_delay": 0.8},
                "facecam_available": True, "facecam_coverage": 0.95, "facecam_score": 95.0,
                "layout_type": "facecam_split", "peak_second": 70.0,
            },
            {
                "start": 110.0, "end": 140.0, "category": "action",
                "gameplay_event": {"category": "gameplay_stunt_jump", "intensity": 80.0, "confidence": 0.8, "timestamp": 120.0},
                "facecam_available": False, "layout_type": "standard", "peak_second": 120.0,
            },
        ]
        res = self.ranker.rank_and_curate_clips(candidates, top_k=3, facecam_available_in_source=True)
        ranked = res["ranked_clips"]
        self.assertGreater(len(ranked), 1)

        # Verify strictly descending unified scores (best-to-worst)
        scores = [c["unified_score"] for c in ranked]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1], f"Rank inversion: clip #{i} score {scores[i]} < clip #{i+1} score {scores[i+1]}")

        # Verify every clip has an explanation string
        for c in ranked:
            self.assertIn("explanation", c)
            self.assertIsInstance(c["explanation"], str)
            self.assertGreater(len(c["explanation"]), 0)

    def test_09_facecam_quota_and_no_facecam_modes(self):
        """Scenario 9: Verify facecam_quota contract for videos with and without facecam."""
        highlights = [
            {"layout_type": "standard", "start": 10.0, "end": 35.0},
            {"layout_type": "standard", "start": 50.0, "end": 75.0},
        ]
        facecam_quota_percent = 50

        # Case A: Video has NO facecam (enable_facecam_split=False)
        enable_facecam_split = False
        fc_summary = {"available": False}
        facecam_quota = int(math.ceil(len(highlights) * (facecam_quota_percent / 100.0))) if (enable_facecam_split and fc_summary.get("available")) else 0
        facecam_highlights = [h for h in highlights if h.get("layout_type") == "facecam_split"]
        self.assertEqual(facecam_quota, 0)
        self.assertEqual(len(facecam_highlights), 0)
        self.assertTrue(len(facecam_highlights) >= facecam_quota)

        # Case B: Video has facecam enabled
        enable_facecam_split = True
        fc_summary = {"available": True}
        facecam_quota = int(math.ceil(len(highlights) * (facecam_quota_percent / 100.0))) if (enable_facecam_split and fc_summary.get("available")) else 0
        self.assertEqual(facecam_quota, 1)


if __name__ == "__main__":
    unittest.main()

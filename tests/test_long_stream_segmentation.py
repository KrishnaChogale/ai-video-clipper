"""tests/test_long_stream_segmentation.py
Tests logical video segmentation, cross-boundary candidate deduplication, and explainable empty-output diagnostics.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.long_stream_manager import (
    plan_video_segments,
    deduplicate_cross_segment_candidates,
    generate_empty_output_diagnostic,
)


class TestLongStreamSegmentation(unittest.TestCase):

    def test_plan_video_segments_short_video(self):
        # 45 minutes = 2700s -> single segment
        segs = plan_video_segments(2700.0)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["start_sec"], 0.0)
        self.assertEqual(segs[0]["end_sec"], 2700.0)
        self.assertTrue(segs[0]["is_single_segment"])

    def test_plan_video_segments_10_hours(self):
        # 10 hours = 36000s
        segs = plan_video_segments(36000.0, segment_target_sec=1800.0, boundary_overlap_sec=30.0)
        self.assertGreater(len(segs), 18)
        # Check that consecutive segments overlap by exactly 30s
        for i in range(len(segs) - 1):
            s1 = segs[i]
            s2 = segs[i + 1]
            # s2 starts 30s before s1 ends
            self.assertAlmostEqual(s1["end_sec"] - s2["start_sec"], 30.0, delta=0.01)

        # Check total coverage reaches 36000s
        self.assertGreaterEqual(segs[-1]["end_sec"], 36000.0)

    def test_deduplicate_cross_segment_candidates(self):
        # Two candidates overlapping by > 50%
        cand1 = {"start": 1780.0, "end": 1810.0, "clip_score": 75.0, "events": ["action"]}
        cand2 = {"start": 1785.0, "end": 1815.0, "clip_score": 88.0, "events": ["action", "scream"]}
        # Independent third candidate
        cand3 = {"start": 2100.0, "end": 2130.0, "clip_score": 60.0}

        deduped = deduplicate_cross_segment_candidates([cand1, cand2, cand3])
        self.assertEqual(len(deduped), 2)
        # Higher scoring cand2 should be retained in place of cand1
        kept_scores = [c["clip_score"] for c in deduped]
        self.assertIn(88.0, kept_scores)
        self.assertIn(60.0, kept_scores)

    def test_empty_output_diagnostic(self):
        mock_rank_result = {
            "ranked_clips": [],
            "rejected_clips": [{"reason": "Score 48.0 below minimum 55.0"}],
            "duplicate_count": 3,
            "rejection_tally": {"Score below quality gate": 5, "Near-duplicate suppression": 3},
            "score_stats": {"min": 32.0, "max": 51.5, "mean": 42.0},
            "near_misses": [{"start": 100.0, "end": 130.0, "unified_score": 51.5, "rejection_reasons": ["Score below minimum 55.0"]}],
        }
        diag = generate_empty_output_diagnostic(
            mock_rank_result,
            total_candidates=8,
            total_duration_sec=36000.0,
            min_quality_score=55.0,
            facecam_summary={"detected": False, "face_presence_ratio": 0.05},
        )
        self.assertEqual(diag["total_candidates"], 8)
        self.assertEqual(diag["accepted_count"], 0)
        self.assertGreater(len(diag["actionable_recommendations"]), 0)
        # Verify recommendation suggests adjusting threshold
        self.assertTrue(any("threshold" in r.lower() for r in diag["actionable_recommendations"]))


if __name__ == "__main__":
    unittest.main()

"""tests/audit_empty_output_diagnostics.py
Test quality gate and empty-output diagnostic generation when 0 clips pass.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.clip_ranker import ClipRanker
from modules.long_stream_manager import generate_empty_output_diagnostic
from modules.pipeline_checkpoint import PipelineCheckpoint

class TestEmptyOutputDiagnostics(unittest.TestCase):
    def test_zero_clips_diagnostic_generation(self):
        # Create candidate pool with moderate scores (40 - 52)
        candidates = [
            {
                "start": 10.0,
                "end": 35.0,
                "score": 50.0,
                "confidence": 0.6,
                "layout_type": "standard",
                "gameplay_event": {"category": "stunt", "intensity": 45.0, "confidence": 0.6, "timestamp": 20.0},
            },
            {
                "start": 60.0,
                "end": 85.0,
                "score": 52.0,
                "confidence": 0.65,
                "layout_type": "standard",
                "gameplay_event": {"category": "gunfight", "intensity": 50.0, "confidence": 0.7, "timestamp": 70.0},
            },
            {
                "start": 120.0,
                "end": 145.0,
                "score": 42.0,
                "confidence": 0.5,
                "layout_type": "standard",
                "gameplay_event": {"category": "chase", "intensity": 40.0, "confidence": 0.5, "timestamp": 130.0},
            },
        ]

        # Set impossible quality threshold (90.0)
        ranker = ClipRanker(min_quality_score=90.0)
        rank_res = ranker.rank_and_curate_clips(
            candidate_clips=candidates,
            top_k=5,
            whisper_segments=[],
            audio_energy=[{"second": i, "energy": 0.3} for i in range(160)],
            visual_scores=[{"second": i, "visual_score": 10.0, "motion": 15.0} for i in range(160)],
        )

        # 1. Verify 0 clips accepted
        self.assertEqual(len(rank_res["ranked_clips"]), 0)
        self.assertGreater(len(rank_res.get("rejected_clips", [])), 0)

        # 2. Verify score_stats and rejection_tally exist
        stats = rank_res.get("score_stats", {})
        self.assertIn("max", stats)
        self.assertIn("min", stats)
        self.assertIn("mean", stats)
        self.assertLess(stats["max"], 90.0)

        rejection_tally = rank_res.get("rejection_tally", {})
        self.assertTrue(any("quality" in k.lower() or "score" in k.lower() for k in rejection_tally.keys()))

        # 3. Generate empty-output diagnostic
        diag = generate_empty_output_diagnostic(
            rank_result=rank_res,
            total_candidates=len(candidates),
            total_duration_sec=160.0,
            min_quality_score=90.0,
            facecam_summary={"detected": False, "face_presence_ratio": 0.0},
        )

        self.assertEqual(diag["total_candidates"], 3)
        self.assertEqual(diag["accepted_count"], 0)
        self.assertEqual(diag["min_quality_threshold"], 90.0)
        self.assertGreaterEqual(diag["near_misses_count"], 1)
        self.assertGreater(len(diag["actionable_recommendations"]), 0)

        # Verify recommended threshold is actionable
        rec_text = " ".join(diag["actionable_recommendations"])
        self.assertIn("threshold", rec_text.lower())
        print("\n✅ Verified 0-clip diagnostic:")
        print(f"   Evaluated: {diag['total_candidates']} candidates")
        print(f"   Score stats: {diag['score_stats']}")
        print(f"   Rejection tally: {diag['rejection_tally']}")
        print(f"   Near misses: {len(diag['near_misses'])}")
        print(f"   Actionable recommendations: {diag['actionable_recommendations']}")

if __name__ == "__main__":
    unittest.main()

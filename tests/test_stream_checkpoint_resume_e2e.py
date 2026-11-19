"""tests/test_stream_checkpoint_resume_e2e.py
End-to-end verification of checkpoint persistence, crash recovery, and instant resume on real stream footage.
"""
import os
import sys
import time
import shutil
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.pipeline_checkpoint import PipelineCheckpoint
from modules.video_ingestion import probe_video
from modules.audio_analysis import analyze_audio, clear_audio_cache
from modules.speech_recognition import transcribe_audio
from modules.object_detection import analyze_visuals
from modules.facecam import summarize_facecam
from modules.event_classification import classify_events
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.clip_ranker import ClipRanker
from modules.long_stream_manager import generate_empty_output_diagnostic

TEST_VIDEO = os.path.join(config.INPUT_DIR, "321206426716_20261118_165332_live_segment.mkv")


class TestStreamCheckpointResumeE2E(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.isfile(TEST_VIDEO), f"Video missing: {TEST_VIDEO}")
        self.ckpt = PipelineCheckpoint()
        self.job_id = self.ckpt.compute_job_id(TEST_VIDEO)
        # Clear any prior test run for clean test
        job_dir = self.ckpt.get_job_dir(self.job_id)
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)

    def tearDown(self):
        clear_audio_cache()

    def test_full_pipeline_checkpoint_and_resume(self):
        print(f"\n1. Initializing checkpoint job: {self.job_id}")
        manifest = self.ckpt.init_job(TEST_VIDEO, {"test_mode": True})
        self.assertIsNotNone(manifest)

        # ── PASS 1: Fresh run of Steps 1-4 ──
        print("2. Pass 1: Fresh Audio Analysis...")
        t0 = time.perf_counter()
        energy_scores, _, sr = analyze_audio(TEST_VIDEO)
        self.ckpt.save_stage(self.job_id, "audio_analysis", energy_scores)
        t_audio = time.perf_counter() - t0
        print(f"   Audio computed in {t_audio:.2f}s ({len(energy_scores)}s)")

        print("3. Pass 1: Fresh Transcription...")
        t0 = time.perf_counter()
        transcript, words_list = transcribe_audio(TEST_VIDEO, chunk_duration_sec=120.0)
        self.ckpt.save_stage(self.job_id, "transcription", {"transcript": transcript, "words_list": words_list})
        t_trans = time.perf_counter() - t0
        print(f"   Transcription computed in {t_trans:.2f}s ({len(transcript)} segs)")

        print("4. Pass 1: Fast Visual Analysis (first 40s)...")
        t0 = time.perf_counter()
        visual_scores = analyze_visuals(TEST_VIDEO, duration_sec=40.0)
        facecam_summary = summarize_facecam(visual_scores, 1920, 1080)
        self.ckpt.save_stage(self.job_id, "visual_analysis", {"visual_scores": visual_scores, "facecam_summary": facecam_summary})
        t_vis = time.perf_counter() - t0
        print(f"   Visual analysis computed in {t_vis:.2f}s ({len(visual_scores)} frames)")

        # Verify checkpoints exist on disk
        self.assertTrue(self.ckpt.is_stage_completed(self.job_id, "audio_analysis"))
        self.assertTrue(self.ckpt.is_stage_completed(self.job_id, "transcription"))
        self.assertTrue(self.ckpt.is_stage_completed(self.job_id, "visual_analysis"))

        # ── PASS 2: Simulate crash and instant resume ──
        print("\n5. Pass 2: SIMULATING CRASH & RESUME...")
        ckpt_resume = PipelineCheckpoint()

        # Audio resume
        t0 = time.perf_counter()
        self.assertTrue(ckpt_resume.is_stage_completed(self.job_id, "audio_analysis"))
        resumed_energy = ckpt_resume.load_stage(self.job_id, "audio_analysis")
        t_audio_resume = time.perf_counter() - t0
        self.assertEqual(len(resumed_energy), len(energy_scores))
        print(f"   ⚡ Audio restored in {t_audio_resume*1000:.2f} ms (vs {t_audio:.2f}s fresh)")
        self.assertLess(t_audio_resume, 0.10) # under 100 ms

        # Transcription resume
        t0 = time.perf_counter()
        self.assertTrue(ckpt_resume.is_stage_completed(self.job_id, "transcription"))
        resumed_trans = ckpt_resume.load_stage(self.job_id, "transcription")
        t_trans_resume = time.perf_counter() - t0
        self.assertEqual(len(resumed_trans["transcript"]), len(transcript))
        print(f"   ⚡ Transcription restored in {t_trans_resume*1000:.2f} ms (vs {t_trans:.2f}s fresh)")
        self.assertLess(t_trans_resume, 0.10) # under 100 ms

        # Visual resume
        t0 = time.perf_counter()
        self.assertTrue(ckpt_resume.is_stage_completed(self.job_id, "visual_analysis"))
        resumed_vis = ckpt_resume.load_stage(self.job_id, "visual_analysis")
        t_vis_resume = time.perf_counter() - t0
        self.assertEqual(len(resumed_vis["visual_scores"]), len(visual_scores))
        print(f"   ⚡ Visuals restored in {t_vis_resume*1000:.2f} ms (vs {t_vis:.2f}s fresh)")
        self.assertLess(t_vis_resume, 0.10) # under 100 ms

        # Total resume speedup
        speedup = (t_audio + t_trans + t_vis) / max(0.001, (t_audio_resume + t_trans_resume + t_vis_resume))
        print(f"\n🎉 Crash Recovery Speedup: {speedup:.0f}× faster than recalculating from zero!")


if __name__ == "__main__":
    unittest.main()

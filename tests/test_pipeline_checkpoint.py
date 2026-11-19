"""tests/test_pipeline_checkpoint.py
Independent verification of PipelineCheckpoint crash recovery, atomic persistence, and stage state tracking.
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.pipeline_checkpoint import PipelineCheckpoint, STANDARD_STAGES


class TestPipelineCheckpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ckpt_test_")
        self.checkpoint = PipelineCheckpoint(checkpoints_dir=self.temp_dir)
        # Create a mock video file for deterministic testing
        self.mock_video = os.path.join(self.temp_dir, "test_livestream.mkv")
        with open(self.mock_video, "wb") as f:
            f.write(b"MOCK_VIDEO_DATA_" * 1024)

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_deterministic_job_id(self):
        id1 = self.checkpoint.compute_job_id(self.mock_video)
        id2 = self.checkpoint.compute_job_id(self.mock_video)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("job_test_livestream_"))

    def test_init_job_manifest(self):
        manifest = self.checkpoint.init_job(self.mock_video, extra_metadata={"duration": 3600.0})
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["source_filename"], "test_livestream.mkv")
        self.assertEqual(manifest["metadata"]["duration"], 3600.0)
        self.assertTrue(os.path.isfile(os.path.join(self.checkpoint.get_job_dir(self.mock_video), "manifest.json")))

    def test_atomic_stage_save_and_load(self):
        job_id = self.checkpoint.compute_job_id(self.mock_video)
        self.checkpoint.init_job(self.mock_video)

        # Stage 1: Audio analysis
        mock_audio_data = [
            {"second": 0, "energy": 0.45, "loudness": 0.12},
            {"second": 1, "energy": 0.88, "loudness": 0.35},
        ]
        saved = self.checkpoint.save_stage(job_id, "audio_analysis", mock_audio_data, {"sample_rate": 16000})
        self.assertTrue(saved)
        self.assertTrue(self.checkpoint.is_stage_completed(job_id, "audio_analysis"))

        loaded = self.checkpoint.load_stage(job_id, "audio_analysis")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["energy"], 0.88)

    def test_all_standard_stages_cycle(self):
        job_id = self.checkpoint.compute_job_id(self.mock_video)
        self.checkpoint.init_job(self.mock_video)

        sample_data = {
            "video_metadata": {"width": 1920, "height": 1080, "fps": 30.0, "duration": 7200.0},
            "audio_analysis": [{"second": i, "energy": 0.5} for i in range(10)],
            "transcription": [{"start": 0.0, "end": 2.5, "text": "Hello world"}],
            "visual_analysis": [{"second": 0, "motion": 12.0}],
            "events_and_scoring": [{"second": 0, "events": ["speech"]}],
            "candidate_selection": [{"start": 10.0, "end": 40.0, "unified_score": 85.0}],
        }

        for stage, data in sample_data.items():
            self.assertFalse(self.checkpoint.is_stage_completed(job_id, stage))
            self.checkpoint.save_stage(job_id, stage, data)
            self.assertTrue(self.checkpoint.is_stage_completed(job_id, stage))
            loaded = self.checkpoint.load_stage(job_id, stage)
            self.assertEqual(loaded, data)

        summary = self.checkpoint.get_job_summary(job_id)
        self.assertEqual(len(summary["completed_stages"]), len(sample_data))
        self.assertIn("clip_export", summary["pending_stages"])

    def test_stage_invalidation_and_retry(self):
        job_id = self.checkpoint.compute_job_id(self.mock_video)
        self.checkpoint.init_job(self.mock_video)
        self.checkpoint.save_stage(job_id, "transcription", [{"text": "First attempt"}])
        self.assertTrue(self.checkpoint.is_stage_completed(job_id, "transcription"))

        # Invalidate stage
        self.checkpoint.invalidate_stage(job_id, "transcription")
        self.assertFalse(self.checkpoint.is_stage_completed(job_id, "transcription"))
        self.assertIsNone(self.checkpoint.load_stage(job_id, "transcription"))

        # Re-save updated data
        self.checkpoint.save_stage(job_id, "transcription", [{"text": "Second attempt"}])
        self.assertTrue(self.checkpoint.is_stage_completed(job_id, "transcription"))
        self.assertEqual(self.checkpoint.load_stage(job_id, "transcription")[0]["text"], "Second attempt")

    def test_rendered_clip_persistence_and_recovery(self):
        job_id = self.checkpoint.compute_job_id(self.mock_video)
        self.checkpoint.init_job(self.mock_video)

        # Mock an exported clip file
        clip_file = os.path.join(self.temp_dir, "clip_01.mp4")
        with open(clip_file, "wb") as f:
            f.write(b"MOCK_MP4_BYTES_" * 1000)

        self.checkpoint.save_rendered_clip(job_id, 0, "clip_01", clip_file, {"score": 88.5})
        rendered = self.checkpoint.get_rendered_clips(job_id)
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0]["clip_id"], "clip_01")
        self.assertEqual(rendered[0]["clip_index"], 0)

    def test_list_all_jobs(self):
        self.checkpoint.init_job(self.mock_video)
        jobs = self.checkpoint.list_all_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["source_filename"], "test_livestream.mkv")


if __name__ == "__main__":
    unittest.main()

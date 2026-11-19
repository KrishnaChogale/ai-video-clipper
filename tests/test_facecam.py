"""Comprehensive Unit & Integration Tests for Upgraded Facecam Detection & Tracking.

Tests MediaPipe detection, Haar fallback, corrupted/missing assets, adaptive EMA,
multi-face disambiguation, grace periods, and downstream layout consumers.
"""
from __future__ import annotations

import os
import sys
import unittest
import tempfile
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.facecam import FacecamAnalyzer, summarize_facecam, build_facecam_layout
from modules.face_tracker import face_crop_x_expression


class TestFacecamPipeline(unittest.TestCase):
    """Test suite for MediaPipe detection, Haar fallback, and tracking smoothing."""

    @classmethod
    def setUpClass(cls):
        cls.real_model_path = os.path.join(BASE_DIR, "models", "face_landmarker.task")
        cls.sample_img_path = os.path.join(BASE_DIR, "temp", "facecam_sample.jpg")
        cls.still_t1800_path = os.path.join(BASE_DIR, "temp", "facecam_t1800.jpg")
        cls.split_test_vid = os.path.join(BASE_DIR, "temp", "facecam_split_test.mp4")

    def test_01_mediapipe_primary_detection(self):
        """Verify MediaPipe detects faces and returns backward-compatible schema with blendshapes."""
        if not os.path.isfile(self.real_model_path):
            self.skipTest("MediaPipe model asset not found on disk")

        analyzer = FacecamAnalyzer(model_path=self.real_model_path)
        self.assertEqual(analyzer.engine, "mediapipe")

        # Test on real test image
        img = cv2.imread(self.sample_img_path)
        if img is None:
            # Fallback synthetic frame with a drawn circle/patch
            img = np.zeros((360, 640, 3), dtype=np.uint8)

        res = analyzer.analyze(img, inset_only=False)
        self.assertIn("box", res)
        self.assertIn("raw_box", res)
        self.assertIn("motion", res)
        self.assertIn("smile", res)
        self.assertIn("engine", res)
        self.assertIn("blendshapes", res)
        self.assertEqual(res["engine"], "mediapipe")

        if res["box"] is not None:
            self.assertEqual(len(res["box"]), 4)
            self.assertGreater(res["box"][2], 0)
            self.assertGreater(res["box"][3], 0)
            self.assertIsInstance(res["blendshapes"], dict)

    def test_02_haar_fallback_when_model_missing(self):
        """Verify automatic, graceful fallback to OpenCV Haar Cascades when model is missing."""
        missing_path = os.path.join(BASE_DIR, "models", "non_existent_model_12345.task")
        analyzer = FacecamAnalyzer(model_path=missing_path)
        self.assertEqual(analyzer.engine, "haar")

        # Create synthetic face-like contrast frame
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        res = analyzer.analyze(frame)
        self.assertEqual(res["engine"], "haar")
        self.assertIn("box", res)
        self.assertIn("motion", res)
        self.assertIn("smile", res)

    def test_03_corrupted_model_asset_handling(self):
        """Verify corrupted or zero-byte model asset falls back to Haar without crashing."""
        with tempfile.NamedTemporaryFile(suffix=".task", delete=False) as tf:
            tf.write(b"CORRUPTED_NON_TFLITE_BYTES_12345")
            corrupt_path = tf.name

        try:
            analyzer = FacecamAnalyzer(model_path=corrupt_path)
            # Should have caught exception and fallen back to Haar
            self.assertEqual(analyzer.engine, "haar")
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            res = analyzer.analyze(frame)
            self.assertEqual(res["engine"], "haar")
        finally:
            if os.path.isfile(corrupt_path):
                os.remove(corrupt_path)

    def test_04_adaptive_velocity_ema_smoothing(self):
        """Verify adaptive EMA suppresses micro-jitter and responds dynamically to movement."""
        analyzer = FacecamAnalyzer(model_path=self.real_model_path, enable_smoothing=True)

        # Micro-jitter sequence (stationary head with +/- 1.5px noise)
        stationary_inputs = [
            [100.0, 50.0, 80.0, 80.0],
            [101.5, 49.0, 80.5, 79.5],
            [99.0, 51.0, 79.5, 80.5],
            [101.0, 50.5, 80.0, 80.0],
            [100.2, 49.5, 80.2, 79.8],
        ]

        smoothed_boxes = []
        for raw in stationary_inputs:
            smoothed = analyzer._apply_adaptive_ema(raw, width=640, height=360)
            analyzer.smoothed_box = smoothed
            smoothed_boxes.append(smoothed)

        raw_x = [b[0] for b in stationary_inputs]
        smooth_x = [b[0] for b in smoothed_boxes]
        self.assertLess(np.var(smooth_x), np.var(raw_x))

        # Test large intentional movement (jump 100 -> 180)
        fast_move = [180.0, 50.0, 80.0, 80.0]
        smooth_after_move = analyzer._apply_adaptive_ema(fast_move, width=640, height=360)
        # Should adaptively respond by moving significantly toward the new position
        self.assertGreater(smooth_after_move[0], 120.0)

    def test_05_temporary_detection_loss_grace_period(self):
        """Verify lost detection for 1-2 frames holds the box rather than snapping to None."""
        analyzer = FacecamAnalyzer(model_path=self.real_model_path)
        valid_frame = cv2.imread(self.still_t1800_path)
        if valid_frame is None:
            valid_frame = np.ones((360, 640, 3), dtype=np.uint8) * 128

        # Frame 1: Detection
        res1 = analyzer.analyze(valid_frame, inset_only=False)
        if res1["box"] is not None:
            saved_box = list(res1["box"])
            # Frame 2: Sudden blank/occluded frame
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            res2 = analyzer.analyze(blank, inset_only=False)
            self.assertTrue(res2.get("held_frame", False))
            self.assertIsNotNone(res2["box"])
            self.assertAlmostEqual(res2["box"][0], saved_box[0], delta=1.0)

            # Frame 3, 4, 5: Exceed max_lost_frames -> resets
            for _ in range(analyzer.max_lost_frames + 1):
                res_last = analyzer.analyze(blank, inset_only=False)
            self.assertIsNone(res_last["box"])

    def test_06_multiple_faces_persistent_tracking(self):
        """Verify multi-face candidate selection chooses the persistent subject."""
        analyzer = FacecamAnalyzer(model_path=self.real_model_path)
        candidates = [
            {"box": [50, 50, 60, 60], "smile": 0.0, "blendshapes": {}},   # corner streamer
            {"box": [280, 150, 100, 100], "smile": 0.0, "blendshapes": {}} # center game char
        ]
        # First frame: prefers edge/corner inset
        chosen = analyzer._select_persistent_face(candidates, width=640, height=360)
        self.assertEqual(chosen["box"], [50, 50, 60, 60])

        # Subsequent frame: maintains tracking consistency with previous box
        analyzer.previous_box = [52, 51, 60, 60]
        # Swap order to verify it doesn't just pick index 0
        reversed_candidates = list(reversed(candidates))
        chosen2 = analyzer._select_persistent_face(reversed_candidates, width=640, height=360)
        self.assertEqual(chosen2["box"], [50, 50, 60, 60])

    def test_07_downstream_summarize_and_layout_compatibility(self):
        """Verify backward compatibility of summarize_facecam and build_facecam_layout."""
        # Mock visual scores over 10 seconds
        scores = []
        for i in range(10):
            scores.append({
                "second": i,
                "facecam_box": [450 + (i % 2), 60, 80, 80],
                "facecam_motion": 15.0 + i,
                "facecam_smile": 20.0,
                "blendshapes": {"jawOpen": 0.1, "browInnerUp": 0.05}
            })

        summary = summarize_facecam(scores, source_width=1920, source_height=1080)
        self.assertTrue(summary["available"])
        self.assertIn("region", summary)
        self.assertEqual(len(summary["presence"]), 10)
        self.assertEqual(len(summary["reaction"]), 10)
        self.assertGreater(summary["region"]["width"], 0)
        self.assertGreater(summary["region"]["height"], 0)

        # Build layout
        layout = build_facecam_layout(
            scores, start=0.0, end=9.0, source_width=1920, source_height=1080,
            reaction=summary, facecam_percent=45, headline_text="TEST HEADLINE"
        )
        self.assertIsNotNone(layout)
        self.assertIn("region", layout)
        self.assertIn("tracks", layout)
        self.assertIn("overlay", layout)
        self.assertEqual(layout["headline_text"], "TEST HEADLINE")
        self.assertGreater(len(layout["tracks"]), 0)

    def test_08_face_tracker_crop_x_expression(self):
        """Verify face_crop_x_expression generates valid FFmpeg crop expressions."""
        if not os.path.isfile(self.split_test_vid):
            self.skipTest("facecam_split_test.mp4 not found on disk")

        expr = face_crop_x_expression(self.split_test_vid, start_sec=0.0, end_sec=3.0, sample_sec=0.5)
        self.assertIsNotNone(expr)
        self.assertIn("iw-ow", expr)
        self.assertIn("max(0", expr)


if __name__ == "__main__":
    unittest.main()

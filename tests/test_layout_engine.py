"""
tests/test_layout_engine.py
Unit tests verifying Phase 4 Dynamic Split-Screen Layouts, Reaction Zoom,
Animated Captions, Context Headlines, and Branding.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from modules.layout_engine import (
    ContextLayoutSelector,
    ReactionZoomController,
    HeadlineBannerGenerator,
    BrandingManager,
)
from modules.subtitle_generator import generate_clip_ass, censor_profanity


class TestLayoutEngineAndPolish(unittest.TestCase):
    """Test suite for Phase 4 dynamic visual features."""

    def test_01_context_layout_selection(self):
        """Verify 50/50, 40/60, 60/40, and non-facecam standard vertical layouts."""
        # 1. No facecam -> standard vertical
        res_noface = ContextLayoutSelector.select_layout(facecam_available=False)
        self.assertEqual(res_noface["layout_type"], "standard_vertical")
        self.assertEqual(res_noface["facecam_percent"], 0)

        # 2. User preference overrides
        res_5050 = ContextLayoutSelector.select_layout(facecam_available=True, user_preference="50/50")
        self.assertEqual(res_5050["layout_ratio"], "50/50")
        self.assertEqual(res_5050["facecam_percent"], 50)

        res_4060 = ContextLayoutSelector.select_layout(facecam_available=True, user_preference="40/60")
        self.assertEqual(res_4060["layout_ratio"], "40/60")
        self.assertTrue(res_4060["streamer_focused"])

        res_6040 = ContextLayoutSelector.select_layout(facecam_available=True, user_preference="60/40")
        self.assertEqual(res_6040["layout_ratio"], "60/40")
        self.assertTrue(res_6040["gameplay_focused"])

        # 3. Autonomous layout selection from intensity
        res_auto_extreme = ContextLayoutSelector.select_layout(facecam_available=True, reaction_intensity=90.0)
        self.assertEqual(res_auto_extreme["layout_ratio"], "60/40")

        res_auto_high = ContextLayoutSelector.select_layout(facecam_available=True, reaction_intensity=78.0)
        self.assertEqual(res_auto_high["layout_ratio"], "50/50")

        res_auto_mod = ContextLayoutSelector.select_layout(facecam_available=True, reaction_intensity=50.0)
        self.assertEqual(res_auto_mod["layout_ratio"], "40/60")

    def test_02_reaction_zoom_half_sine_curve(self):
        """Verify reaction zoom generates smooth half-sine expressions and suppresses noise."""
        reactions = [
            {"start": 12.0, "end": 14.5, "category": "streamer_hype", "confidence": 0.85},
        ]
        zoom_res = ReactionZoomController.build_zoom_expressions(
            reaction_events=reactions,
            gameplay_events=[],
            start_sec=10.0,
            end_sec=25.0,
            enabled=True,
            max_reaction_zoom=0.12,
        )
        face_expr = zoom_res["face_zoom_expr"]
        self.assertIsNotNone(face_expr)
        self.assertIn("1.0+", face_expr)
        self.assertIn("sin(PI*(t-", face_expr)
        self.assertIn("between(t,", face_expr)

        # Mathematical verification: at boundary (t=2.0s relative), zoom must equal 1.0 (sin(0)=0)
        # Parse expression parameters
        self.assertIn("between(t,2.000,4.500)", face_expr)

        # Verify low-confidence reaction is ignored
        low_reactions = [
            {"start": 12.0, "end": 14.5, "category": "streamer_hype", "confidence": 0.40},
        ]
        zoom_low = ReactionZoomController.build_zoom_expressions(
            reaction_events=low_reactions,
            gameplay_events=[],
            start_sec=10.0,
            end_sec=25.0,
            enabled=True,
            min_reaction_conf=0.65,
        )
        self.assertIsNone(zoom_low["face_zoom_expr"])

        # Verify disabled zoom returns None
        zoom_disabled = ReactionZoomController.build_zoom_expressions(
            reaction_events=reactions,
            gameplay_events=[],
            start_sec=10.0,
            end_sec=25.0,
            enabled=False,
        )
        self.assertIsNone(zoom_disabled["face_zoom_expr"])

    def test_03_gameplay_climax_zoom(self):
        """Verify gameplay explosions and gunfights generate subtle gameplay punch-in."""
        gameplay = [
            {"timestamp": 15.0, "category": "gameplay_explosion", "confidence": 0.88},
        ]
        zoom_res = ReactionZoomController.build_zoom_expressions(
            reaction_events=[],
            gameplay_events=gameplay,
            start_sec=10.0,
            end_sec=25.0,
            enabled=True,
        )
        game_expr = zoom_res["game_zoom_expr"]
        self.assertIsNotNone(game_expr)
        self.assertIn("1.0+", game_expr)
        self.assertIn("sin(PI*", game_expr)

    def test_04_context_headline_generation(self):
        """Verify dynamic, authentic headline generation across diverse events."""
        # Surprise / Gunfight
        h_surprise = HeadlineBannerGenerator.generate_headline(
            gameplay_event="gameplay_gunfight",
            reaction_category="streamer_surprise",
            speech_text="no way bro",
            intensity=80.0,
        )
        self.assertIn("EXPECT", h_surprise["headline"])
        self.assertEqual(h_surprise["accent_color"], "0x00F0FF")  # Electric cyan

        # Hype / Kill
        h_hype = HeadlineBannerGenerator.generate_headline(
            gameplay_event="gameplay_kill_death",
            reaction_category="streamer_hype",
            speech_text="let's go!",
            intensity=92.0,
        )
        self.assertTrue("CLUTCH" in h_hype["headline"] or "MASTERCLASS" in h_hype["headline"])
        self.assertEqual(h_hype["accent_color"], "0xFFE600")  # Neon yellow

        # Scream / Chaos
        h_scream = HeadlineBannerGenerator.generate_headline(
            gameplay_event="gameplay_collision_crash",
            reaction_category="streamer_scream",
            intensity=85.0,
        )
        self.assertTrue("LIFE" in h_scream["headline"] or "ACCIDENT" in h_scream["headline"])
        self.assertEqual(h_scream["accent_color"], "0xFF3366")  # Crimson

        # Anticipation
        h_ant = HeadlineBannerGenerator.generate_headline(
            gameplay_event="gameplay_gunfight",
            reaction_category="streamer_gasp",
            reaction_delay=-1.5,
        )
        self.assertEqual(h_ant["headline"], "HE KNEW WHAT WAS COMING!")

    def test_05_animated_captions_karaoke_and_safe_zone(self):
        """Verify word-level kinetic scale, emotion pop, profanity masking, and split safe margin."""
        words = [
            {"start": 1.0, "end": 1.3, "word": "bro"},
            {"start": 1.3, "end": 1.7, "word": "fucking"},
            {"start": 1.7, "end": 2.2, "word": "INSANE!"},
        ]
        # Generate with facecam_split=True
        ass_split = generate_clip_ass(
            words_list=words,
            clip_start=0.0,
            clip_end=4.0,
            crop_vertical=True,
            dynamic_highlight=True,
            censor_swear_words=True,
            facecam_split=True,
        )
        # 1. Safe zone vertical margin for split-screen (780px from bottom)
        self.assertIn(",780,1", ass_split)

        # 2. Profanity masked
        self.assertIn("F*CKING", ass_split)
        self.assertNotIn("FUCKING", ass_split)

        # 3. Kinetic scale tags on active words
        self.assertIn(r"\fscx112\fscy112", ass_split)
        self.assertIn(r"\fscx116\fscy116", ass_split)  # High-arousal INSANE pop!

        # 4. Standard vertical (non-facecam) margin remains 440px
        ass_standard = generate_clip_ass(
            words_list=words,
            clip_start=0.0,
            clip_end=4.0,
            crop_vertical=True,
            dynamic_highlight=True,
            facecam_split=False,
        )
        self.assertIn(",440,1", ass_standard)

    def test_06_branding_configuration(self):
        """Verify configurable branding, watermark opacity, and position."""
        brand = BrandingManager.get_branding_config(custom_branding="@GamerFleetLive")
        self.assertEqual(brand["branding_text"], "@GamerFleetLive")
        self.assertEqual(brand["watermark_text"], "@GamerFleetLive")
        self.assertGreaterEqual(brand["opacity"], 0.2)
        self.assertLessEqual(brand["opacity"], 1.0)
        self.assertEqual(brand["position"], "bottom_right")


if __name__ == "__main__":
    unittest.main()

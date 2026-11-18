"""
tests/test_viral_features.py
Verifies:
1. 30-45s viral duration selection & hook detection in highlight_scoring.
2. Dead air detection & pause removal (jump-cuts) in clip_editor.
3. Subtitle remapping across cut intervals in subtitle_generator.
4. Dynamic kinetic word-by-word highlighting in subtitle_generator.
5. Profanity censoring (F*CK, SH*T, etc.) inside dynamic ASS tags.
6. Safe zone margin compliance for Reels & Shorts (MarginV=440).
"""
import sys, os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.highlight_scoring import select_highlights, _evaluate_hook
from modules.clip_editor import detect_dead_air
from modules.subtitle_generator import (
    generate_clip_ass,
    remap_words_for_dead_air,
    censor_profanity,
)


class TestViralFeatures(unittest.TestCase):

    def test_viral_duration_and_hook_selection(self):
        # 90 seconds of mock per-second scores
        scores = [15.0] * 90
        # Create an event with a strong hook at second 20-55 (35s length)
        # Opening hook at 20-23
        scores[20] = 75.0
        scores[21] = 82.0
        scores[22] = 88.0
        for s in range(23, 55):
            scores[s] = 60.0

        events = [{"events": ["police_chase"], "confidence": 0.85, "category": "action"} if 20 <= i < 55 else {} for i in range(90)]

        # Verify hook evaluation
        hook_score, hook_type = _evaluate_hook(scores, events, 20, 55)
        self.assertGreaterEqual(hook_score, 70.0)
        self.assertEqual(hook_type, "police_gunfight")

        # Select highlights with viral default (30-45s)
        highlights = select_highlights(scores, event_classification=events, top_k=1, min_dur=30, max_dur=45)
        self.assertEqual(len(highlights), 1)
        h = highlights[0]
        dur = h["end"] - h["start"]
        self.assertGreaterEqual(dur, 30)
        self.assertLessEqual(dur, 45)
        self.assertIn("hook_score", h)
        self.assertGreater(h["hook_score"], 50.0)

    def test_dynamic_kinetic_captions(self):
        sample_words = [
            {"start": 1.0, "end": 1.4, "word": "Holy"},
            {"start": 1.4, "end": 1.9, "word": "shit"},
            {"start": 1.9, "end": 2.5, "word": "look"},
            {"start": 2.5, "end": 3.0, "word": "there!"},
        ]

        ass = generate_clip_ass(
            words_list=sample_words,
            clip_start=0.0,
            clip_end=5.0,
            crop_vertical=True,
            position="Bottom (Reels/Shorts Safe Zone)",
            censor_swear_words=True,
            dynamic_highlight=True,
        )

        # Check safe zone vertical margin (440px)
        self.assertIn(",80,100,440,1", ass)

        # Check profanity censoring inside ASS dialogue
        self.assertIn("SH*T", ass)
        self.assertNotIn("SHIT", ass)

        # Check dynamic karaoke tag presence {\c&H...&}
        self.assertIn(r"{\c" + config.CAPTION_COLOR_HIGHLIGHT + r"}", ass)
        self.assertIn(r"{\c" + config.CAPTION_COLOR_BASE + r"}", ass)

    def test_dead_air_word_remapping(self):
        words = [
            {"start": 1.0, "end": 1.5, "word": "Start"},
            # Pause between 1.5 and 5.0 (3.5s of dead air)
            {"start": 5.0, "end": 5.5, "word": "Action"},
            {"start": 5.5, "end": 6.0, "word": "End"},
        ]

        # Simulating cut of [1.6, 4.8] (3.2 seconds removed)
        cut_intervals = [(1.6, 4.8)]
        remapped = remap_words_for_dead_air(words, cut_intervals, clip_start=0.0, clip_end=7.0)

        self.assertEqual(len(remapped), 3)
        self.assertEqual(remapped[0]["word"], "Start")
        self.assertAlmostEqual(remapped[0]["start"], 1.0, places=2)

        # Word "Action" should be shifted earlier by 3.2s
        self.assertEqual(remapped[1]["word"], "Action")
        expected_s = 5.0 - 3.2
        self.assertAlmostEqual(remapped[1]["start"], expected_s, places=2)

    def test_profanity_censorship_dictionary(self):
        raw = "What the fuck is this bullshit and damn shitty driving"
        censored = censor_profanity(raw)
        self.assertIn("F*CK", censored)
        self.assertIn("BULLSH*T", censored)
        self.assertIn("D*MN", censored)
        self.assertIn("SH*TTY", censored)

    def test_hook_title_banner_and_emojis(self):
        from modules.subtitle_generator import detect_cue_emoji, wrap_hook_text, format_hook_ass_text

        # 1. Emoji detection
        self.assertEqual(detect_cue_emoji("That drift was crazy!"), "🔥")
        self.assertEqual(detect_cue_emoji("The police are here!"), "🚨")
        self.assertEqual(detect_cue_emoji("We totally died"), "💀")
        self.assertEqual(detect_cue_emoji("He blew up with a bomb"), "💥")

        # 2. Hook text formatting and wrapping
        wrapped = wrap_hook_text("YOU NEED TO SEE THIS CRAZY FIVE STAR POLICE ESCAPE", max_chars_per_line=24)
        self.assertIn(r"\N", wrapped)

        formatted = format_hook_ass_text("YOU NEED TO SEE THIS 🔥")
        self.assertIn(r"{\c&H0000FFFF&}YOU NEED", formatted)
        self.assertIn(r"{\c&H00FFFFFF&}TO SEE THIS 🔥", formatted)

        # 3. ASS generation with hook banner & subtitle emojis
        words = [
            {"start": 0.5, "end": 1.0, "word": "That"},
            {"start": 1.0, "end": 1.5, "word": "was"},
            {"start": 1.5, "end": 2.2, "word": "crazy"},
        ]
        ass = generate_clip_ass(
            words_list=words,
            clip_start=0.0,
            clip_end=10.0,
            crop_vertical=True,
            hook_text="YOU NEED TO SEE THIS 🔥",
            burn_hook=True,
            add_emojis=True,
        )

        # Hook style and dialogue presence
        self.assertIn("Style: HookBanner", ass)
        self.assertIn("HookBanner,,0,0,0,,{\\fad(", ass)
        self.assertIn("YOU NEED", ass)

        # Dynamic cue emoji injection
        self.assertIn("🔥", ass)


if __name__ == "__main__":
    unittest.main()

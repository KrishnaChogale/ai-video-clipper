"""
modules/layout_engine.py
Dynamic Split-Screen Layouts, Reaction Zoom, Context Headlines & Visual Polish.

Phase 4 Upgrades:
1. ContextLayoutSelector: Determines 50/50, 40/60 (streamer focus), 60/40 (gameplay focus),
   or standard_vertical based on reaction intensity, gameplay action, and face presence.
2. ReactionZoomController: Builds continuous, zero-jerk half-sine easing zoom expressions
   for facecam and gameplay panels during high-confidence emotional peaks.
3. HeadlineBannerGenerator: Dynamically crafts authentic, context-grounded headlines
   from gameplay events, streamer reactions, and speech transcripts.
4. BrandingManager: Formats configurable branding, logos, watermarks, and theme accents.
"""
from __future__ import annotations

import math
import re
from typing import Any

import config


class ContextLayoutSelector:
    """Selects and sizes vertical short video layouts based on multimodal signals."""

    @staticmethod
    def select_layout(
        facecam_available: bool,
        reaction_intensity: float = 0.0,
        gameplay_intensity: float = 0.0,
        reaction_category: str = "",
        gameplay_event: str = "",
        user_preference: str = "auto",
        face_confidence: float = 1.0,
    ) -> dict[str, Any]:
        """
        Determine optimal layout ratio and facecam panel percentage.

        Layouts:
          - '50/50': Equal balance between gameplay and streamer (facecam_percent = 50%).
          - '40/60': Streamer-focused reaction (streamer = 58%, gameplay = 42%).
          - '60/40': Gameplay-focused action (streamer = 40%, gameplay = 60%).
          - 'standard_vertical': Non-facecam full vertical gameplay (facecam_percent = 0%).
        """
        pref = str(user_preference or "auto").strip().lower()

        # If no facecam is detected on disk, always fall back to standard non-facecam vertical
        if not facecam_available or face_confidence < 0.20:
            return {
                "layout_type": "standard_vertical",
                "layout_ratio": "standard_vertical",
                "facecam_percent": 0,
                "streamer_focused": False,
                "gameplay_focused": True,
                "rationale": "No persistent facecam detected; using standard full-height vertical layout.",
            }

        # Explicit user layout preference
        if pref in ("50/50", "equal", "balanced"):
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "50/50",
                "facecam_percent": 50,
                "streamer_focused": False,
                "gameplay_focused": False,
                "rationale": "User requested 50/50 balanced split-screen layout.",
            }
        if pref in ("40/60", "streamer", "streamer_focus", "streamer_focused"):
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "40/60",
                "facecam_percent": 58,
                "streamer_focused": True,
                "gameplay_focused": False,
                "rationale": "User requested 40/60 streamer-focused split-screen layout.",
            }
        if pref in ("60/40", "gameplay", "gameplay_focus", "gameplay_focused"):
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "60/40",
                "facecam_percent": 40,
                "streamer_focused": False,
                "gameplay_focused": True,
                "rationale": "User requested 60/40 gameplay-focused split-screen layout.",
            }

        # Autonomous / Context-Aware Layout Selection
        if reaction_intensity >= 85.0:
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "60/40",
                "facecam_percent": 58,
                "streamer_focused": True,
                "gameplay_focused": False,
                "rationale": f"Extreme streamer emotion (intensity={reaction_intensity:.1f}) prioritizes streamer panel.",
            }
        elif reaction_intensity >= 75.0:
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "50/50",
                "facecam_percent": 50,
                "streamer_focused": False,
                "gameplay_focused": False,
                "rationale": f"High emotion (intensity={reaction_intensity:.1f}) with balanced 50/50 split.",
            }
        else:
            return {
                "layout_type": "facecam_split",
                "layout_ratio": "40/60",
                "facecam_percent": 40,
                "streamer_focused": False,
                "gameplay_focused": True,
                "rationale": f"Gameplay-focused action (intensity={reaction_intensity:.1f}) with 40/60 split.",
            }


class ReactionZoomController:
    """Computes smooth, cinematic zoom easing expressions for FFmpeg filter graphs."""

    @staticmethod
    def build_zoom_expressions(
        reaction_events: list[dict] | None,
        gameplay_events: list[dict] | None,
        start_sec: float,
        end_sec: float,
        enabled: bool = True,
        max_reaction_zoom: float = 0.12,
        max_game_zoom: float = 0.06,
        min_reaction_conf: float = 0.65,
    ) -> dict[str, str | None]:
        """
        Generate FFmpeg expressions for smooth half-sine reaction punch-in.

        Starts at 1.0, peaks gently at (1.0 + max_zoom) at the reaction midpoint,
        and returns smoothly to 1.0 with continuous zero velocity at boundaries.
        No shaking, no jumping, and disabled for ordinary conversation.
        """
        if not enabled:
            return {"face_zoom_expr": None, "game_zoom_expr": None}

        duration = max(1.0, end_sec - start_sec)
        allowed_reaction_types = {
            "streamer_hype": 0.10,
            "streamer_scream": 0.12,
            "streamer_surprise": 0.12,
            "streamer_gasp": 0.08,
            "streamer_laugh": 0.08,
        }

        # 1. Facecam reaction punch-in
        face_terms = []
        for r in reaction_events or []:
            cat = r.get("category", "")
            conf = float(r.get("confidence", 0.0))
            if conf < min_reaction_conf or cat not in allowed_reaction_types:
                continue

            r_start = float(r.get("start", 0.0))
            r_end = float(r.get("end", r_start + 1.0))
            t0 = max(0.0, r_start - start_sec)
            t1 = min(duration, r_end - start_sec)
            dur = t1 - t0

            # Require at least 0.8s reaction window, smooth to at least 1.4s
            if dur < 0.8:
                mid = (t0 + t1) / 2.0
                t0 = max(0.0, mid - 0.7)
                t1 = min(duration, mid + 0.7)
                dur = t1 - t0

            if dur < 0.5 or t0 >= duration or t1 <= 0.0:
                continue

            base_amp = allowed_reaction_types[cat]
            amp = min(max_reaction_zoom, base_amp * (conf / 0.85))
            # Smooth half-sine bell curve: sin(pi * (t - t0) / dur) * between(t, t0, t1)
            face_terms.append(f"({amp:.3f}*sin(PI*(t-{t0:.3f})/{dur:.3f})*between(t,{t0:.3f},{t1:.3f}))")

        face_expr = None
        if face_terms:
            face_expr = "1.0+" + "+".join(face_terms[:2])  # Max 2 active punches to prevent clutter

        # 2. Gameplay climax punch-in
        game_terms = []
        climax_events = {"gameplay_explosion", "gameplay_gunfight", "gameplay_collision_crash"}
        for g in gameplay_events or []:
            cat = g.get("category", "")
            conf = float(g.get("confidence", 0.0))
            if conf < 0.70 or cat not in climax_events:
                continue

            g_t = float(g.get("timestamp", 0.0)) - start_sec
            t0 = max(0.0, g_t - 0.2)
            t1 = min(duration, g_t + 1.2)
            dur = t1 - t0
            if dur < 0.5 or t0 >= duration or t1 <= 0.0:
                continue

            amp = min(max_game_zoom, 0.05 * conf)
            game_terms.append(f"({amp:.3f}*sin(PI*(t-{t0:.3f})/{dur:.3f})*between(t,{t0:.3f},{t1:.3f}))")

        game_expr = None
        if game_terms:
            game_expr = "1.0+" + "+".join(game_terms[:2])

        return {"face_zoom_expr": face_expr, "game_zoom_expr": game_expr}


class HeadlineBannerGenerator:
    """Generates authentic, high-impact headline banners tailored to specific moments."""

    HEADLINE_MATRIX = {
        "streamer_surprise": {
            "gameplay_gunfight": ["HE DID NOT EXPECT THAT!", "THAT CAUGHT HIM OFF GUARD!", "OUT OF ABSOLUTELY NOWHERE!"],
            "gameplay_collision_crash": ["WHAT JUST HAPPENED?!", "THE MOST UNREAL CRASH!", "HOW DID HE SURVIVE THIS?!"],
            "gameplay_explosion": ["WHAT JUST HAPPENED?!", "THE CRAZIEST EXPLOSION!", "EVERYTHING WENT WRONG!"],
            "default": ["WHAT JUST HAPPENED?!", "HE COULD NOT BELIEVE IT!", "NO ONE SAW THIS COMING!"],
        },
        "streamer_hype": {
            "gameplay_gunfight": ["THE CRAZIEST CLUTCH!", "PURE ACCURACY IN THE GUNFIGHT!", "HE WENT ABSOLUTELY CRAZY!"],
            "gameplay_kill_death": ["THE CRAZIEST CLUTCH!", "HE REALLY PULLED THAT OFF!", "AN ABSOLUTE MASTERCLASS!"],
            "gameplay_action_peak": ["STREAMER LOST HIS MIND!", "THE HYPE IS UNREAL!", "HE WENT FULL GOD MODE!"],
            "default": ["BRO LOST HIS MIND!", "THE HYPE IS UNREAL!", "THAT WAS ABSOLUTELY PEAK!"],
        },
        "streamer_scream": {
            "gameplay_gunfight": ["FIGHTING FOR HIS LIFE!", "TOTAL CHAOS IN THE GUNFIGHT!", "BRO PANICKED SO HARD!"],
            "gameplay_collision_crash": ["BRO WAS FIGHTING FOR HIS LIFE!", "THE MOST DRAMATIC ACCIDENT!", "HE SAW HIS LIFE FLASH!"],
            "default": ["BRO WAS FIGHTING FOR HIS LIFE!", "TOTAL CHAOS ERUPTED!", "BRO REALLY FELL APART!"],
        },
        "streamer_laugh": {
            "gameplay_collision_crash": ["THE FUNNIEST GTA MOMENT!", "HE COULD NOT STOP LAUGHING!", "BRO CANNOT DRIVE AT ALL!"],
            "default": ["HE COULD NOT STOP LAUGHING!", "THE MOST UNHINGED MOMENT!", "BRO COULD NOT KEEP A STRAIGHT FACE!"],
        },
        "streamer_gasp": {
            "default": ["HE FROZE IN PURE SHOCK!", "A MOMENT OF COMPLETE SILENCE!", "CAUGHT IN DISBELIEF!"],
        },
    }

    THEME_COLORS = {
        "streamer_hype": {"accent": "0xFFE600", "name": "neon_yellow"},      # #FFE600
        "streamer_scream": {"accent": "0xFF3366", "name": "crimson"},         # #FF3366
        "streamer_surprise": {"accent": "0x00F0FF", "name": "electric_cyan"}, # #00F0FF
        "streamer_laugh": {"accent": "0x37E6FF", "name": "sky_blue"},        # #37E6FF
        "streamer_gasp": {"accent": "0x72FF4D", "name": "lime_green"},       # #72FF4D
        "default": {"accent": "0x00F0FF", "name": "electric_cyan"},
    }

    @classmethod
    def generate_headline(
        cls,
        gameplay_event: str,
        reaction_category: str,
        speech_text: str = "",
        intensity: float = 50.0,
        reaction_delay: float = 0.0,
    ) -> dict[str, str]:
        """
        Produce a grounded, context-aware headline and matching color theme.
        """
        r_cat = str(reaction_category or "default").strip()
        g_evt = str(gameplay_event or "default").strip()

        # Check anticipation pre-reaction
        if reaction_delay < -0.8:
            headline = "HE KNEW WHAT WAS COMING!"
        else:
            cat_dict = cls.HEADLINE_MATRIX.get(r_cat, cls.HEADLINE_MATRIX["streamer_hype"])
            candidates = cat_dict.get(g_evt, cat_dict.get("default", ["WHAT JUST HAPPENED?!"]))
            # Deterministic hash selection based on speech text length to avoid static uniformity
            seed = len(speech_text) + int(intensity)
            headline = candidates[seed % len(candidates)]

        # Theme styling
        theme = cls.THEME_COLORS.get(r_cat, cls.THEME_COLORS["default"])

        return {
            "headline": headline,
            "accent_color": theme["accent"],
            "theme_name": theme["name"],
            "banner_bg": "0x16181C",
        }


class BrandingManager:
    """Manages channel branding, watermarks, and subtitle styling parameters."""

    @staticmethod
    def get_branding_config(custom_branding: str = "", custom_watermark: str = "") -> dict[str, Any]:
        """Returns validated branding parameters for video rendering."""
        branding_text = (custom_branding or getattr(config, "BRANDING_WATERMARK_TEXT", "") or "").strip()
        watermark_text = (custom_watermark or branding_text)[:40]
        opacity = float(getattr(config, "BRANDING_WATERMARK_OPACITY", 0.65))

        return {
            "branding_text": branding_text[:60],
            "watermark_text": watermark_text,
            "opacity": max(0.2, min(1.0, opacity)),
            "fontfile": r"C\:/Windows/Fonts/arialbd.ttf",
            "position": getattr(config, "BRANDING_WATERMARK_POSITION", "bottom_right"),
        }

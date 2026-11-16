"""
modules/event_classification.py
Combines audio + visual signals into per-second GTA gameplay event tags.
Pure heuristics — no extra model loading.
"""
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

CATEGORIES = ["action", "funny", "cinematic", "discovery", "dialogue", "unknown"]


def classify_events(
    energy_buckets: list[dict],
    visual_scores: list[dict],
    transcript: list[dict],
) -> list[dict]:
    """
    For each second, return an event dict with:
      { second, category, events: [str], bonus_score: float }
    """
    total = min(len(energy_buckets), len(visual_scores))
    if total == 0:
        return []

    # Build speech density per second
    speech_density = [0.0] * total
    for seg in transcript:
        wps = len(seg["text"].split()) / max(1.0, seg["end"] - seg["start"])
        for s in range(int(seg["start"]), min(total, int(seg["end"]) + 1)):
            speech_density[s] += wps

    # Normalise visual motion
    max_motion = max((v["motion"] for v in visual_scores[:total]), default=1.0) or 1.0

    classified = []
    for sec in range(total):
        ab = energy_buckets[sec]
        vb = visual_scores[sec]
        events   = []
        bonus    = 0.0
        category = "unknown"

        # ── Audio events ──
        if ab.get("is_explosion"):
            events.append("explosion")
            bonus += config.GTA_EVENT_BONUS["explosion"]
        if ab.get("is_gunshot") and not ab.get("is_explosion"):
            events.append("gunshot")
            bonus += config.GTA_EVENT_BONUS["gunshot"]

        # ── Visual events ──
        motion_norm = vb["motion"] / max_motion
        if motion_norm > 0.75:
            events.append("high_speed")
            bonus += config.GTA_EVENT_BONUS["high_speed"]
        if vb.get("scene_change"):
            events.append("scene_change")
            bonus += 5

        # ── Combined heuristics ──
        loudness = ab["loudness"]
        spike    = ab["spike"]

        if ("explosion" in events or "gunshot" in events) and motion_norm > 0.5:
            events.append("police_chase")
            bonus += config.GTA_EVENT_BONUS["police_chase"]

        if motion_norm > 0.6 and vb["visual_score"] > 3.0:
            events.append("crash")
            bonus += config.GTA_EVENT_BONUS["crash"]

        if motion_norm > 0.85 and vb["visual_score"] > 2.0:
            events.append("stunt")
            bonus += config.GTA_EVENT_BONUS["stunt"]

        # ── Dialogue / NPC interaction ──
        if speech_density[sec] > 2.5:
            events.append("dialogue")
            if spike > 5.0:
                events.append("npc_reaction")
                bonus += config.GTA_EVENT_BONUS["npc_reaction"]

        # ── Cinematic ── (low motion, visually rich but no explosions)
        if ("explosion" not in events and "gunshot" not in events
                and motion_norm < 0.3 and vb["visual_score"] > 1.5):
            events.append("cinematic")
            bonus += config.GTA_EVENT_BONUS["cinematic"]

        # ── Assign primary category ──
        if any(e in events for e in ("explosion", "police_chase", "gunshot", "crash", "stunt")):
            category = "action"
        elif "npc_reaction" in events:
            category = "funny"
        elif "cinematic" in events:
            category = "cinematic"
        elif "dialogue" in events:
            category = "dialogue"
        elif "scene_change" in events:
            category = "discovery"

        classified.append({
            "second":       sec,
            "category":     category,
            "events":       list(set(events)),
            "bonus_score":  round(bonus, 2),
        })

    return classified

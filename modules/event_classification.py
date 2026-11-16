"""Conservative event evidence aggregation.

These signals are proxies, not GTA event recognition. Semantic labels are only
emitted when evidence is corroborated; unsupported moments remain unknown.
"""
import logging

logger = logging.getLogger(__name__)


def classify_events(energy_buckets, visual_scores, transcript):
    total = min(len(energy_buckets), len(visual_scores))
    if not total:
        return []
    speech = [False] * total
    for seg in transcript or []:
        for sec in range(max(0, int(seg.get("start", 0))), min(total, int(seg.get("end", 0)) + 1)):
            speech[sec] = True
    motion = [float(v.get("motion", 0)) for v in visual_scores[:total]]
    # Local robust baseline distinguishes sustained motion from one-frame flashes.
    baseline = sorted(motion)[int((len(motion) - 1) * .70)] if motion else 0
    result = []
    for sec in range(total):
        a, v = energy_buckets[sec], visual_scores[sec]
        events = []
        confidence = 0.0
        temporal = 0.0
        start, end = max(0, sec - 2), min(total, sec + 3)
        run = motion[start:end]
        sustained = len(run) >= 3 and sum(x > max(8.0, baseline * .75) for x in run) >= 3
        if sustained:
            events.append("sustained_motion")
            temporal = min(1.0, sum(run) / max(1, len(run)) / 60.0)
        # Heuristic audio flags are retained as weak clues only. Never translate
        # them into named weapons/explosions/chases without an audio classifier.
        if a.get("is_gunshot") or a.get("is_explosion"):
            events.append("possible_audio_transient")
        if speech[sec]:
            events.append("speech")
        if v.get("scene_change"):
            events.append("scene_change")
        # No present model distinguishes collisions, pursuits, stunts, NPC comedy,
        # thefts, or cinematic intent. Do not claim those categories from proxies.
        category = "unknown"
        result.append({"second": sec, "category": category, "events": events,
                       "confidence": round(confidence, 2),
                       "temporal_strength": round(temporal, 2),
                       "bonus_score": 0.0})
    return result

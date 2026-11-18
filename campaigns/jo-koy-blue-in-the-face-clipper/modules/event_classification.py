"""Stand-up clip cues based on transcript markers and audio timing.

Only explicit laughter transcription is treated as a crowd-reaction label.
Energy/onset combinations remain low-confidence possible-reaction cues.
"""
import re

LAUGHTER_RE = re.compile(r"\b(laugh(?:s|ing|ter)?|haha+|ha ha+|audience laughs|crowd laughs)\b", re.I)


def classify_events(energy_buckets, visual_scores, transcript):
    total = min(len(energy_buckets), len(visual_scores))
    if not total:
        return []
    speech = [False] * total
    laughter = [False] * total
    for seg in transcript or []:
        start = max(0, int(float(seg.get("start", 0))))
        end = min(total, int(float(seg.get("end", 0))) + 1)
        text = seg.get("text", "")
        for sec in range(start, end):
            speech[sec] = True
            if LAUGHTER_RE.search(text):
                laughter[sec] = True

    loudness = [float(x.get("loudness", 0)) for x in energy_buckets[:total]]
    spikes = [float(x.get("spike", 0)) for x in energy_buckets[:total]]
    motion = [float(x.get("motion", 0)) for x in visual_scores[:total]]
    loud_cut = sorted(loudness)[int((total - 1) * 0.75)]
    spike_cut = sorted(spikes)[int((total - 1) * 0.85)]
    motion_cut = sorted(motion)[int((total - 1) * 0.75)]
    out = []
    for sec in range(total):
        events, category, confidence, temporal = [], "unknown", 0.0, 0.0
        if speech[sec]:
            events.append("speech")
        if speech[sec] and motion[sec] >= motion_cut:
            events.append("performance_motion_candidate")
        if laughter[sec]:
            events.extend(["explicit_laughter_cue", "crowd_reaction"])
            category, confidence, temporal = "crowd_reaction", 0.85, 0.8
        elif (loudness[sec] >= loud_cut and spikes[sec] >= spike_cut and
              any(speech[max(0, sec - 4):sec])):
            events.append("possible_audience_reaction")
            confidence, temporal = 0.35, 0.35
        if visual_scores[sec].get("scene_change"):
            events.append("scene_change")
        out.append({"second": sec, "category": category, "events": events,
                    "confidence": confidence, "temporal_strength": temporal,
                    "bonus_score": 0.0})
    return out

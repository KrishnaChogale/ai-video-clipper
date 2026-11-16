"""
modules/highlight_scoring.py
Multi-modal highlight scoring (0–100) with configurable weights.
Selects non-overlapping clips and respects content-style preference.
Supports multi-scale dynamic window durations (min_dur to max_dur).
"""
import logging
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise a list to [0, 1]."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def compute_per_second_scores(
    energy_buckets: list[dict],
    visual_scores: list[dict],
    transcript: list[dict],
    event_classification: list[dict] = None,
    weights: dict | None = None,
) -> list[float]:
    """
    Returns a float score per second in [0, 100].
    """
    if weights is None:
        weights = config.SCORE_WEIGHTS

    total = min(len(energy_buckets), len(visual_scores))
    if event_classification:
        total = min(total, len(event_classification))

    if total == 0:
        return []

    # Raw signals
    loudness = [energy_buckets[s]["loudness"]   for s in range(total)]
    spike    = [energy_buckets[s]["spike"]       for s in range(total)]
    visual   = [visual_scores[s]["visual_score"] for s in range(total)]
    motion   = [visual_scores[s]["motion"]       for s in range(total)]
    
    if event_classification:
        bonus = [event_classification[s].get("bonus_score", 0.0) for s in range(total)]
    else:
        bonus = [0.0] * total

    # Build speech density
    speech_density = [0.0] * total
    for seg in (transcript or []):
        wps = len(seg["text"].split()) / max(1.0, seg["end"] - seg["start"])
        for s in range(int(seg["start"]), min(total, int(seg["end"]) + 1)):
            speech_density[s] += wps

    # Normalise each signal independently
    n_loud   = _normalise(loudness)
    n_spike  = _normalise(spike)
    n_visual = _normalise(visual)
    n_motion = _normalise(motion)
    n_speech = _normalise(speech_density)
    n_bonus  = _normalise(bonus)

    w = weights
    total_weight = sum(w.values()) or 100

    scores = []
    for s in range(total):
        # visual_action = YOLO score + motion
        visual_action    = (n_visual[s] * 0.6 + n_motion[s] * 0.4) * 100
        # audio_excitement = loudness + spike
        audio_excitement = (n_loud[s] * 0.5 + n_spike[s] * 0.5) * 100
        # unexpected_event = bonus from GTA event classifier
        unexpected_event = n_bonus[s] * 100
        # story_context = speech density
        story_context    = n_speech[s] * 100
        # event_clarity — higher when visual score is high and motion is moderate
        event_clarity    = (n_visual[s] * 0.7 + (1.0 - abs(n_motion[s] - 0.5)) * 0.3) * 100
        # payoff_potential — favours spikey moments
        payoff_potential = n_spike[s] * 100

        raw = (
            visual_action    * w.get("visual_action",    25) +
            audio_excitement * w.get("audio_excitement", 20) +
            unexpected_event * w.get("unexpected_event", 20) +
            story_context    * w.get("story_context",    15) +
            event_clarity    * w.get("event_clarity",    10) +
            payoff_potential * w.get("payoff_potential", 10)
        ) / total_weight

        scores.append(round(min(100.0, raw), 2))

    return scores


def select_highlights(
    per_second_scores: list[float],
    event_classification: list[dict] = None,
    top_k: int = 5,
    window_sec: int = 30,
    min_dur: int = None,
    max_dur: int = None,
    min_score: float = 10.0,
    style: str = "balanced",
    excluded_categories: list[str] | None = None,
) -> list[dict]:
    """
    Peak detection returning top non-overlapping highlight windows.
    Supports fixed window_sec or dynamic range between min_dur and max_dur.
    Returns list of dicts: {start, end, score, category, events, peak_second}
    """
    total = len(per_second_scores)
    if total == 0:
        return []

    event_classification = event_classification or []
    excluded = set(excluded_categories or [])
    cat_map   = {e["second"]: e.get("category", "unknown") for e in event_classification}
    event_map = {e["second"]: e.get("events", []) for e in event_classification}

    # Determine candidate window sizes
    if min_dur and max_dur and min_dur != max_dur:
        window_sizes = sorted(list({min_dur, (min_dur + max_dur) // 2, max_dur}))
    else:
        window_sizes = [window_sec if not min_dur else min_dur]

    candidates = []
    for w_size in window_sizes:
        if total < w_size:
            w_size = max(1, total)

        for start in range(0, max(1, total - w_size + 1), 2): # step by 2s for efficiency
            end = min(total, start + w_size)
            window = per_second_scores[start:end]
            if not window:
                continue
            avg_score = sum(window) / len(window)
            peak_sec  = start + window.index(max(window))

            # Style filtering modifier
            cats_in_window = {cat_map.get(s, "unknown") for s in range(start, end)}
            style_lower = (style or "balanced").lower()
            if style_lower == "action" and "action" not in cats_in_window:
                avg_score *= 0.65
            elif style_lower == "cinematic" and "cinematic" not in cats_in_window:
                avg_score *= 0.65
            elif style_lower == "funny" and "funny" not in cats_in_window:
                avg_score *= 0.65

            if excluded and cats_in_window.issubset(excluded):
                continue

            dom_cat = "unknown"
            if event_classification:
                cats_counter = Counter(cat_map.get(s, "unknown") for s in range(start, end))
                if cats_counter:
                    dom_cat = cats_counter.most_common(1)[0][0]

            events = list({ev for s in range(start, end) for ev in event_map.get(s, [])})

            candidates.append({
                "start":        start,
                "end":          end,
                "score":        round(avg_score, 2),
                "category":     dom_cat,
                "events":       events,
                "peak_second":  peak_sec,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Greedy non-overlapping selection
    selected = []
    for cand in candidates:
        if cand["score"] < min_score:
            continue
        # Check overlap with already chosen clips
        overlaps = any(
            not (cand["end"] <= ch["start"] or cand["start"] >= ch["end"])
            for ch in selected
        )
        if not overlaps:
            selected.append(cand)
        if len(selected) >= min(top_k, config.CLIP_TOP_K_MAX):
            break

    selected.sort(key=lambda x: x["start"])
    logger.info("Highlight selection: %d clips selected", len(selected))
    return selected

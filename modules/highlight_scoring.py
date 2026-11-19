"""Conservative temporal highlight ranking using the project's local signals.

Scores are evidence summaries, not social-platform predictions. Candidate windows
are grown around sustained activity and corroborated event signals.
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def _bounded(value, scale):
    return max(0.0, min(1.0, float(value or 0.0) / scale))


def _evaluate_hook(per_second_scores, events, start, end, hook_window_sec=3):
    """
    Evaluates viral hook strength in the first 3 seconds of a candidate window.
    High retention on Reels/Shorts demands immediate motion, audio spike, or event trigger.
    Returns (hook_score: float [0-100], hook_type: str).
    """
    hook_end = min(end, start + hook_window_sec)
    if hook_end <= start:
        return 0.0, "none"

    hook_slice = per_second_scores[start:hook_end]
    avg_hook_energy = sum(hook_slice) / len(hook_slice) if hook_slice else 0.0

    hook_events = set()
    for sec in range(start, min(len(events), hook_end)):
        hook_events.update(events[sec].get("events", []))

    hook_type = "action_onset"
    bonus = 0.0
    if any(e in hook_events for e in ["explosion", "crash", "stunt"]):
        hook_type = "high_impact_visual"
        bonus += 16.0
    elif any(e in hook_events for e in ["police_chase", "gunshot"]):
        hook_type = "police_gunfight"
        bonus += 12.0
    elif any(e in hook_events for e in ["npc_reaction", "high_speed"]):
        hook_type = "speed_chaos"
        bonus += 8.0

    hook_score = min(100.0, max(0.0, avg_hook_energy * 0.85 + bonus))
    return round(hook_score, 1), hook_type


def compute_per_second_scores(energy_buckets, visual_scores, transcript,
                             event_classification=None, weights=None):
    """Score evidence with fixed saturation points; avoid per-video min/max inflation."""
    total = min(len(energy_buckets), len(visual_scores))
    if event_classification:
        total = min(total, len(event_classification))
    if not total:
        return []
    speech = [0.0] * total
    for seg in transcript or []:
        duration = max(1.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
        density = len(seg.get("text", "").split()) / duration
        for sec in range(max(0, int(seg.get("start", 0))), min(total, int(seg.get("end", 0)) + 1)):
            speech[sec] += density

    scores = []
    for sec in range(total):
        a, v = energy_buckets[sec], visual_scores[sec]
        ev = event_classification[sec] if event_classification else {}
        motion = _bounded(v.get("motion"), 45)
        objects = _bounded(v.get("visual_score"), 8)
        loud = _bounded(a.get("loudness"), 0.35)
        transient = _bounded(a.get("spike"), 12)
        speech_ctx = _bounded(speech[sec], 3)
        corroborated = _bounded(ev.get("confidence", 0), 1)
        # Loudness and transients have a capped 15% combined contribution.
        factor = {
            "event_significance": corroborated,
            "event_clarity": min(objects, motion),
            "visual_action": 0.55 * motion + 0.45 * objects,
            "audio_excitement": 0.6 * loud + 0.4 * transient,
            "unexpectedness": corroborated,
            "build_up_payoff": _bounded(ev.get("temporal_strength", 0), 1),
            "story_context": speech_ctx,
            "replay_potential": 0.6 * corroborated + 0.4 * motion,
            "video_quality": 1.0 - (0.5 if v.get("scene_change") else 0.0),
            "visual_beauty": _bounded(v.get("scenic_visual_score"), 100),
        }
        w = weights or config.HIGHLIGHT_FACTOR_WEIGHTS
        factor_total = sum(max(0.0, float(w.get(k, 0))) for k in factor) or 100.0
        score = 100.0 * sum(factor[k] * max(0.0, float(w.get(k, 0))) for k in factor) / factor_total
        if motion < 0.12 and objects < 0.25 and corroborated < 0.3:
            score -= 8.0  # Static footage receives an explicit boredom penalty.
        # Save component evidence for explainable candidate ranking.
        scores.append(round(max(0.0, min(100.0, score)), 2))
    return scores


def select_highlights(per_second_scores, event_classification=None, top_k=5,
                      window_sec=30, min_dur=None, max_dur=None, min_score=10.0,
                      style="balanced", excluded_categories=None,
                      excluded_ranges=None, facecam_scores=None,
                      facecam_presence=None, require_facecam=False,
                      min_facecam_score=0.0, min_facecam_coverage=0.60):
    """Rank automatically generated windows around local activity peaks.

    The app must cut clips without asking the user to supply timestamps. This
    searches windows within the user's duration bounds, rewards sustained
    activity, penalizes dead time, and greedily removes overlapping picks.
    """
    n = len(per_second_scores)
    if not n:
        return []
    events = event_classification or []
    low = max(1, int(min_dur if min_dur is not None else getattr(config, "CLIP_VIRAL_MIN_DURATION", window_sec)))
    high = max(low, int(max_dur if max_dur is not None else getattr(config, "CLIP_VIRAL_MAX_DURATION", window_sec)))
    max_len = min(n, high)
    min_len = min(max_len, low)
    excluded = set(excluded_categories or [])
    excluded_ranges = excluded_ranges or []
    baseline = sorted(per_second_scores)[int((n - 1) * 0.45)]
    active_cutoff = max(baseline + 2.0, sorted(per_second_scores)[int((n - 1) * 0.70)])
    # Evaluate the configured range in 5-second increments. Choosing only the
    # minimum/midpoint/maximum made shorter windows disproportionately likely.
    window_sizes = list(range(min_len, max_len + 1, 5))
    if not window_sizes or window_sizes[-1] != max_len:
        window_sizes.append(max_len)
    candidates = []
    for length in window_sizes:
        if length > n:
            continue
        step = max(1, min(3, length // 10))
        starts = list(range(0, n - length + 1, step))
        if starts and starts[-1] != n - length:
            starts.append(n - length)
        for start in starts:
            end = start + length
            if any(start < float(r_end) and end > float(r_start)
                   for r_start, r_end in excluded_ranges):
                continue
            window = per_second_scores[start:end]
            mean_score = sum(window) / length
            peak = max(window)
            coverage = sum(x >= active_cutoff for x in window) / length
            above_baseline = sum(max(0.0, x - baseline) for x in window) / length
            # The strongest continuous activity run estimates how much context
            # the event needs. A 35-second developing event should not be forced
            # into a 15-second cut just because its peak is concentrated.
            peak_index = start + window.index(peak)
            event_start = event_end = peak_index
            while event_start > 0 and per_second_scores[event_start - 1] >= active_cutoff:
                event_start -= 1
            while event_end + 1 < n and per_second_scores[event_end + 1] >= active_cutoff:
                event_end += 1
            event_span = event_end - event_start + 1
            target_length = min(max_len, max(min_len, event_span + 10))
            duration_fit = 1.0 - abs(length - target_length) / max(1, max_len - min_len)
            relevant = events[start:min(end, len(events))]
            labels = {label for e in relevant for label in e.get("events", [])}
            cat_scores = {}
            confs = []
            for e in relevant:
                cat = e.get("category", "unknown")
                cat_scores[cat] = cat_scores.get(cat, 0) + 1
                confs.append(float(e.get("confidence", 0)))
            category = max(cat_scores, key=cat_scores.get) if cat_scores else "unknown"
            if category in excluded:
                continue
            # A peak contributes, but cannot carry a mostly inactive window.
            quality_score = (0.58 * mean_score + 0.18 * peak + 14 * coverage +
                             0.55 * above_baseline)
            face_score = 0.0
            face_coverage = 0.0
            if facecam_scores is not None and facecam_presence is not None:
                fs = facecam_scores[start:min(end, len(facecam_scores))]
                fp = facecam_presence[start:min(end, len(facecam_presence))]
                face_score = sum(float(x) for x in fs) / max(1, len(fs))
                face_coverage = sum(1 for x in fp if x) / max(1, len(fp))
            if require_facecam:
                if face_coverage < min_facecam_coverage or face_score < min_facecam_score:
                    continue
                # Keep gameplay/audio quality meaningful while prioritizing
                # windows where the streamer's face shows a reaction.
                quality_score = 0.42 * quality_score + 0.58 * face_score
            if quality_score < min_score:
                continue

            # Evaluate opening 3-second hook for immediate short-form viewer retention
            hook_score, hook_type = _evaluate_hook(per_second_scores, events, start, end)

            # Viral formula: 40% quality & action, 35% duration fit, 25% opening hook
            score = 0.40 * quality_score + 0.35 * (100 * duration_fit) + 0.25 * hook_score
            if style.lower() in {"action", "funny", "cinematic"} and category == style.lower():
                score += 2.0
            if style.lower().replace(" ", "_") in {"scenic", "scenic_showcase"} and category == "scenic":
                score += 10.0
            confidence = sum(confs) / len(confs) if confs else 0.0
            candidates.append({"start": start, "end": end, "score": round(score, 2),
                               "category": category, "events": sorted(labels),
                               "peak_second": start + window.index(peak),
                               "confidence": round(confidence, 2),
                               "hook_score": hook_score,
                               "hook_type": hook_type,
                               "facecam_score": round(face_score, 2),
                               "facecam_coverage": round(face_coverage, 2),
                               "layout_type": "facecam_split" if require_facecam else "standard",
                               "reason": f"Ranked with {hook_type} hook ({hook_score:.0f}%) and sustained action",
                               "score_factors": {"window_mean": round(mean_score, 2),
                                                 "quality_score": round(quality_score, 2),
                                                 "hook_score": hook_score,
                                                 "activity_coverage": round(coverage, 2),
                                                 "peak_score": round(peak, 2),
                                                 "target_duration": target_length,
                                                 "duration_fit": round(duration_fit, 2),
                                                 "event_confidence": round(confidence, 2)}})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    for c in candidates:
        if c["score"] < min_score:
            continue
        if any(not (c["end"] <= x["start"] or c["start"] >= x["end"]) for x in selected):
            continue
        selected.append(c)
        if len(selected) >= min(top_k, config.CLIP_TOP_K_MAX):
            break
    selected.sort(key=lambda c: c["start"])
    logger.info("Temporal highlight selection: %d candidates", len(selected))
    return selected

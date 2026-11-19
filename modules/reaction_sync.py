"""modules/reaction_sync.py
Phase 3: Reaction-to-Gameplay Synchronization Engine.

Connects gameplay events (audio transients, visual motion bursts, transcript cues)
with Phase 2 streamer reactions (laughter, screaming, gasping, hype, etc.).

Features:
- Evidence-based gameplay event detection (explosions, gunfights, crashes, kill/death).
- Adaptive temporal alignment with asymmetric windows:
  * Anticipation: -3.0s to 0.0s (reaction before gameplay trigger).
  * Immediate reaction: 0.0s to 1.8s (natural physiological reaction time).
  * Delayed commentary: +1.8s to +5.0s (reaction after processing gameplay event).
- Explainable synchronization scoring.
- Complete 4-part highlight construction: Setup -> Trigger -> Reaction -> Payoff.
- Redundancy and overlap elimination via temporal Non-Maximum Suppression (NMS).
- Dynamic facecam layout recommendations (50/50, 40/60, 60/40, standard vertical).
- 100% CPU inference (0 MB GPU VRAM overhead).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

GAMEPLAY_CATEGORIES = [
    "gameplay_explosion",
    "gameplay_gunfight",
    "gameplay_collision_crash",
    "gameplay_kill_death",
    "gameplay_action_peak",
    "gameplay_sudden_transient",
    "unknown",
]


class GameplayEventDetector:
    """Detects candidate gameplay events from audio, visual motion, and transcript evidence."""

    def __init__(self):
        self.kill_death_keywords = [
            "kill", "killed", "dead", "died", "headshot", "wasted", "busted",
            "eliminated", "sniper", "rpg", "mara gaya", "khatam", "shot him", "got him"
        ]
        self.action_keywords = [
            "explosion", "crashed", "police", "chase", "helicopter", "tank", "stunt"
        ]

    def detect_events(
        self,
        energy_buckets: list[dict[str, Any]],
        visual_scores: list[dict[str, Any]],
        whisper_segments: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Identify timestamped gameplay events from energy buckets and visual scores.
        
        Returns:
            list of dicts: {
                timestamp, category, confidence, intensity, evidence
            }
        """
        total_sec = min(len(energy_buckets), len(visual_scores)) if visual_scores else len(energy_buckets)
        if total_sec == 0:
            return []

        # Index speech text by second
        speech_by_second: dict[int, str] = {}
        if whisper_segments:
            for seg in whisper_segments:
                start_s = int(seg.get("start", 0))
                end_s = int(seg.get("end", 0))
                txt = seg.get("text", "").lower()
                for s in range(start_s, end_s + 1):
                    speech_by_second[s] = (speech_by_second.get(s, "") + " " + txt).strip()

        # Compute rolling motion baseline and motion delta
        motions = [float(v.get("motion", 0.0)) for v in visual_scores[:total_sec]] if visual_scores else [0.0] * total_sec
        motion_baseline = float(np_median(motions)) if motions else 0.0

        candidates = []

        for sec in range(total_sec):
            a_item = energy_buckets[sec] if sec < len(energy_buckets) else {}
            v_item = visual_scores[sec] if visual_scores and sec < len(visual_scores) else {}
            speech_txt = speech_by_second.get(sec, "")

            loudness = float(a_item.get("loudness", 0.0))
            spike = float(a_item.get("spike", 0.0))
            flatness = float(a_item.get("spectral_flatness", 0.0))
            is_explosion = bool(a_item.get("is_explosion", False))
            is_gunshot = bool(a_item.get("is_gunshot", False))

            motion = float(v_item.get("motion", 0.0))
            prev_motion = motions[sec - 1] if sec > 0 else motion
            motion_delta = max(0.0, motion - prev_motion)
            scene_change = bool(v_item.get("scene_change", False))

            has_kd_word = any(kw in speech_txt for kw in self.kill_death_keywords)
            has_action_word = any(kw in speech_txt for kw in self.action_keywords)

            category = "unknown"
            confidence = 0.0
            intensity = 0.0

            # 1. Gameplay Explosion / Heavy Blast
            if is_explosion or (loudness > 0.18 and flatness > 0.22 and spike > 8.0):
                category = "gameplay_explosion"
                confidence = min(0.95, 0.50 + (0.25 if is_explosion else 0.10) + min(0.20, loudness))
                intensity = min(100.0, loudness * 250.0 + spike * 2.0)

            # 2. Gameplay Gunfight / Rapid Shooting
            elif is_gunshot or (spike > 11.0 and loudness > 0.10 and flatness > 0.10):
                category = "gameplay_gunfight"
                confidence = min(0.90, 0.45 + (0.25 if is_gunshot else 0.10) + min(0.20, spike / 30.0))
                intensity = min(100.0, spike * 4.5 + loudness * 150.0)

            # 3. Vehicle Collision / Hard Crash
            elif (motion_delta > 18.0 or (motion > 35.0 and scene_change)) and (spike > 6.0 or loudness > 0.08):
                category = "gameplay_collision_crash"
                confidence = min(0.85, 0.40 + min(0.25, motion_delta / 40.0) + (0.20 if spike > 8.0 else 0.0))
                intensity = min(100.0, motion * 1.5 + spike * 3.0)

            # 4. Kill / Death Trigger
            elif has_kd_word and (loudness > 0.08 or motion > 15.0 or scene_change):
                category = "gameplay_kill_death"
                confidence = min(0.85, 0.50 + (0.20 if scene_change else 0.10) + (0.15 if spike > 8.0 else 0.0))
                intensity = min(100.0, 50.0 + loudness * 150.0 + motion)

            # 5. General High-Intensity Action Peak
            elif motion > 32.0 and loudness > 0.12 and spike > 7.0:
                category = "gameplay_action_peak"
                confidence = min(0.80, 0.35 + min(0.25, motion / 60.0) + min(0.20, loudness / 0.30))
                intensity = min(100.0, motion * 1.2 + loudness * 120.0)

            # 6. Sudden Audio Transient
            elif spike > 14.0 and loudness > 0.06:
                category = "gameplay_sudden_transient"
                confidence = min(0.70, 0.30 + min(0.30, spike / 30.0))
                intensity = min(100.0, spike * 4.0)

            if category != "unknown" and confidence >= 0.35:
                candidates.append({
                    "timestamp": float(sec),
                    "category": category,
                    "confidence": round(confidence, 3),
                    "intensity": round(intensity, 1),
                    "evidence": {
                        "loudness": round(loudness, 4),
                        "spike": round(spike, 2),
                        "flatness": round(flatness, 4),
                        "is_explosion": is_explosion,
                        "is_gunshot": is_gunshot,
                        "motion": round(motion, 1),
                        "motion_delta": round(motion_delta, 1),
                        "scene_change": scene_change,
                        "speech_keyword": has_kd_word or has_action_word,
                    }
                })

        # Cluster contiguous events of the same category (keep peak)
        clustered = self._cluster_gameplay_events(candidates)
        return clustered

    def _cluster_gameplay_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cluster temporally adjacent gameplay triggers within 2 seconds into a single peak trigger."""
        if not events:
            return []
        
        clustered = []
        current_cluster = [events[0]]

        for ev in events[1:]:
            prev = current_cluster[-1]
            if ev["category"] == prev["category"] and (ev["timestamp"] - prev["timestamp"]) <= 2.0:
                current_cluster.append(ev)
            else:
                # Finalize current cluster: choose event with highest confidence * intensity
                best = max(current_cluster, key=lambda x: x["confidence"] * x["intensity"])
                clustered.append(best)
                current_cluster = [ev]

        if current_cluster:
            best = max(current_cluster, key=lambda x: x["confidence"] * x["intensity"])
            clustered.append(best)

        return clustered


class AdaptiveReactionSynchronizer:
    """Synchronizes detected gameplay events with streamer reactions and constructs highlight clips."""

    def __init__(
        self,
        min_delay: float = -3.0,
        max_delay: float = 5.0,
        optimal_delay_min: float = 0.2,
        optimal_delay_max: float = 1.8,
        min_sync_confidence: float = 0.40,
        default_pre_roll: float = 4.5,
        default_post_roll: float = 4.5,
        min_clip_duration: float = 15.0,
        max_clip_duration: float = 60.0,
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.optimal_delay_min = optimal_delay_min
        self.optimal_delay_max = optimal_delay_max
        self.min_sync_confidence = min_sync_confidence
        self.default_pre_roll = default_pre_roll
        self.default_post_roll = default_post_roll
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration

    def compute_temporal_alignment(
        self,
        delay: float,
        is_speech_continuous: bool = False,
        has_anticipation_word: bool = False,
    ) -> float:
        """Calculate explainable temporal alignment score (0.0 to 1.0) based on reaction delay.
        
        Args:
            delay: (reaction_timestamp - gameplay_timestamp) in seconds.
            is_speech_continuous: True if streamer continues talking between gameplay and reaction.
            has_anticipation_word: True if anticipation words ("watch this", "wait") preceded gameplay.
        """
        # Outside acceptable window: rejected
        if delay < self.min_delay or delay > self.max_delay:
            return 0.0

        # 1. Optimal physiological reaction window (+0.2s to +1.8s)
        if self.optimal_delay_min <= delay <= self.optimal_delay_max:
            return 1.0

        # 2. Immediate simultaneous reaction (0.0s to 0.2s)
        if 0.0 <= delay < self.optimal_delay_min:
            return 0.90 + 0.10 * (delay / self.optimal_delay_min)

        # 3. Delayed reaction / Commentary (+1.8s to +5.0s)
        if delay > self.optimal_delay_max:
            # Smooth linear decline from 1.0 down to 0.35 at max_delay
            fraction = (delay - self.optimal_delay_max) / (self.max_delay - self.optimal_delay_max)
            base_score = 1.0 - 0.65 * fraction
            if is_speech_continuous:
                base_score = min(1.0, base_score + 0.20)  # Commentary bridges the delay
            return max(0.20, min(1.0, base_score))

        # 4. Anticipation / Pre-reaction (-3.0s to 0.0s)
        # Streamer reacts before the gameplay trigger happens (e.g. anticipates crash or failure)
        anticipation_fraction = abs(delay) / abs(self.min_delay)
        ant_score = 0.85 - 0.45 * anticipation_fraction
        if has_anticipation_word:
            ant_score = min(1.0, ant_score + 0.25)
        return max(0.20, min(1.0, ant_score))

    def evaluate_pairing(
        self,
        gameplay_event: dict[str, Any],
        reaction_event: dict[str, Any],
        speech_text: str = "",
    ) -> dict[str, Any]:
        """Score an individual candidate (gameplay, reaction) pair."""
        t_game = float(gameplay_event["timestamp"])
        t_react = float(reaction_event["start"])
        delay = round(t_react - t_game, 2)

        # Speech flags
        speech_lower = speech_text.lower()
        has_ant_word = any(w in speech_lower for w in ["watch", "wait", "look", "dekh", "dhyan se", "here we go"])
        is_speech_cont = len(speech_text.strip().split()) >= 3

        # Temporal alignment
        temp_score = self.compute_temporal_alignment(
            delay=delay,
            is_speech_continuous=is_speech_cont,
            has_anticipation_word=has_ant_word,
        )

        if temp_score <= 0.0:
            return {
                "status": "unmatched",
                "reason": f"Delay {delay}s outside window [{self.min_delay}s, {self.max_delay}s]",
                "delay": delay,
            }

        g_conf = float(gameplay_event.get("confidence", 0.0))
        r_conf = float(reaction_event.get("confidence", 0.0))
        g_cat = gameplay_event.get("category", "")
        r_cat = reaction_event.get("category", "")

        # Cross-modal audio/event coherence bonus
        coherence = 0.0
        # High impact action + high arousal reaction
        if g_cat in ["gameplay_explosion", "gameplay_collision_crash"] and r_cat in ["streamer_scream", "streamer_gasp", "streamer_laugh"]:
            coherence += 0.20
        elif g_cat == "gameplay_kill_death" and r_cat in ["streamer_hype", "streamer_laugh", "streamer_surprise"]:
            coherence += 0.20
        elif g_cat == "gameplay_gunfight" and r_cat in ["streamer_hype", "streamer_scream"]:
            coherence += 0.15

        # Synchronization Confidence Formula
        sync_conf = (
            0.28 * g_conf
            + 0.32 * r_conf
            + 0.25 * temp_score
            + 0.15 * min(1.0, coherence + (0.15 if is_speech_cont else 0.0))
        )
        sync_conf = round(min(1.0, sync_conf), 3)

        status = "matched" if sync_conf >= self.min_sync_confidence else "unmatched"

        return {
            "status": status,
            "gameplay_event": g_cat,
            "gameplay_timestamp": t_game,
            "reaction_category": r_cat,
            "reaction_timestamp": t_react,
            "reaction_delay": delay,
            "synchronization_confidence": sync_conf,
            "evidence": {
                "temporal_alignment": round(temp_score, 3),
                "gameplay_confidence": g_conf,
                "reaction_confidence": r_conf,
                "cross_modal_coherence": round(coherence, 2),
                "speech_continuity": is_speech_cont,
                "gameplay_evidence": gameplay_event.get("evidence", {}),
                "reaction_evidence": reaction_event.get("evidence", {}),
            }
        }

    def construct_highlight_boundaries(
        self,
        t_game: float,
        t_react_start: float,
        t_react_end: float,
        total_duration: float,
        transcript_segments: list[dict[str, Any]] | None = None,
    ) -> tuple[float, float]:
        """Construct complete 4-part highlight: Setup -> Trigger -> Reaction -> Payoff."""
        trigger_time = min(t_game, t_react_start)
        reaction_finish = max(t_game, t_react_end)

        # 1. Adaptive Setup (Pre-roll)
        pre_roll = self.default_pre_roll
        # If speech started slightly earlier, align to sentence boundary
        if transcript_segments:
            for seg in transcript_segments:
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                if (trigger_time - 7.0) <= s_start <= (trigger_time - 1.0):
                    pre_roll = max(pre_roll, trigger_time - s_start + 0.5)
                    break

        clip_start = max(0.0, round(trigger_time - pre_roll, 1))

        # 2. Adaptive Payoff (Post-roll)
        post_roll = self.default_post_roll
        # If streamer speech continues after reaction, let them complete sentence
        if transcript_segments:
            for seg in transcript_segments:
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                if reaction_finish <= s_start <= (reaction_finish + 4.0):
                    post_roll = max(post_roll, s_end - reaction_finish + 1.0)
                    break

        clip_end = min(total_duration, round(reaction_finish + post_roll, 1))

        # Enforce duration bounds
        dur = clip_end - clip_start
        if dur < self.min_clip_duration:
            deficit = self.min_clip_duration - dur
            # Expand pre-roll and post-roll evenly
            clip_start = max(0.0, round(clip_start - deficit * 0.5, 1))
            clip_end = min(total_duration, round(clip_start + self.min_clip_duration, 1))
        elif dur > self.max_clip_duration:
            clip_end = round(clip_start + self.max_clip_duration, 1)

        return clip_start, clip_end

    def synchronize(
        self,
        gameplay_events: list[dict[str, Any]],
        reaction_events: list[dict[str, Any]],
        total_video_duration: float,
        transcript_segments: list[dict[str, Any]] | None = None,
        facecam_scores: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Pair all gameplay and reaction events, score synchronization, and build clips with NMS.
        
        Returns:
            list of matched, deduplicated highlight dicts ordered by quality score.
        """
        candidate_matches = []
        used_reaction_ids = set()

        for g_idx, g_ev in enumerate(gameplay_events):
            t_g = float(g_ev["timestamp"])

            # Find matching reaction events within the synchronization window
            best_pair = None
            best_sync_conf = -1.0

            for r_idx, r_ev in enumerate(reaction_events):
                t_r = float(r_ev["start"])
                delay = t_r - t_g
                if delay < self.min_delay or delay > self.max_delay:
                    continue

                # Context speech text between t_g and t_r
                window_speech = ""
                if transcript_segments:
                    w_start = min(t_g, t_r) - 1.0
                    w_end = max(t_g, t_r + r_ev.get("duration", 1.0)) + 1.0
                    texts = [s.get("text", "") for s in transcript_segments if s.get("end", 0) >= w_start and s.get("start", 0) <= w_end]
                    window_speech = " ".join(texts)

                pair_res = self.evaluate_pairing(g_ev, r_ev, speech_text=window_speech)
                if pair_res["status"] == "matched" and pair_res["synchronization_confidence"] > best_sync_conf:
                    best_sync_conf = pair_res["synchronization_confidence"]
                    best_pair = (pair_res, r_ev, r_idx)

            if best_pair is not None:
                pair_res, r_ev, r_idx = best_pair
                used_reaction_ids.add(r_idx)

                # Construct complete highlight boundaries
                c_start, c_end = self.construct_highlight_boundaries(
                    t_game=pair_res["gameplay_timestamp"],
                    t_react_start=r_ev["start"],
                    t_react_end=r_ev["end"],
                    total_duration=total_video_duration,
                    transcript_segments=transcript_segments,
                )

                # Layout Recommendation based on facecam presence in this window
                layout_rec = self._recommend_layout(c_start, c_end, facecam_scores, r_ev)

                # Overall clip quality score combining sync confidence and intensity
                total_intensity = max(float(g_ev.get("intensity", 0.0)), float(r_ev.get("intensity", 0.0)))
                overall_score = round(pair_res["synchronization_confidence"] * 55.0 + total_intensity * 0.45, 1)

                candidate_matches.append({
                    "gameplay_event": pair_res["gameplay_event"],
                    "gameplay_timestamp": pair_res["gameplay_timestamp"],
                    "reaction_category": pair_res["reaction_category"],
                    "reaction_timestamp": pair_res["reaction_timestamp"],
                    "reaction_delay": pair_res["reaction_delay"],
                    "synchronization_confidence": pair_res["synchronization_confidence"],
                    "clip_score": overall_score,
                    "clip_start": c_start,
                    "clip_end": c_end,
                    "clip_duration": round(c_end - c_start, 1),
                    "layout_type": layout_rec["layout_type"],
                    "layout_ratio": layout_rec["layout_ratio"],
                    "evidence": pair_res["evidence"],
                    "status": "matched",
                })

        # Apply Non-Maximum Suppression (NMS) to eliminate overlapping and duplicate clips
        deduped = self._apply_nms(candidate_matches)
        return deduped

    def _recommend_layout(
        self,
        start_sec: float,
        end_sec: float,
        facecam_scores: list[dict[str, Any]] | None,
        reaction_event: dict[str, Any],
    ) -> dict[str, str]:
        """Determine layout type (facecam_split vs standard_vertical) and split ratio."""
        if not facecam_scores:
            return {"layout_type": "standard_vertical", "layout_ratio": "none"}

        # Check facecam coverage in [start_sec, end_sec]
        s_idx = int(start_sec)
        e_idx = int(end_sec)
        slice_fc = [f for f in facecam_scores if s_idx <= f.get("second", -1) <= e_idx]

        if not slice_fc:
            return {"layout_type": "standard_vertical", "layout_ratio": "none"}

        valid_boxes = sum(1 for f in slice_fc if f.get("facecam_box") is not None or f.get("box") is not None)
        coverage = valid_boxes / max(1, len(slice_fc))

        # If coverage is sufficient (> 40%), delegate to ContextLayoutSelector
        if coverage >= 0.40:
            from modules.layout_engine import ContextLayoutSelector
            layout_res = ContextLayoutSelector.select_layout(
                facecam_available=True,
                reaction_intensity=float(reaction_event.get("intensity", 50.0)),
                reaction_category=str(reaction_event.get("category", "")),
                face_confidence=coverage,
            )
            return {"layout_type": layout_res["layout_type"], "layout_ratio": layout_res["layout_ratio"]}

        return {"layout_type": "standard_vertical", "layout_ratio": "none"}

    def _apply_nms(self, clips: list[dict[str, Any]], iou_threshold: float = 0.35) -> list[dict[str, Any]]:
        """Greedy Non-Maximum Suppression to eliminate overlapping highlight clips."""
        if not clips:
            return []

        # Sort clips by clip_score descending
        sorted_clips = sorted(clips, key=lambda c: c["clip_score"], reverse=True)
        selected = []

        for c in sorted_clips:
            # Check overlap against all already selected clips
            keep = True
            for sel in selected:
                # Intersection over Union
                inter_start = max(c["clip_start"], sel["clip_start"])
                inter_end = min(c["clip_end"], sel["clip_end"])
                inter_dur = max(0.0, inter_end - inter_start)

                union_dur = (c["clip_end"] - c["clip_start"]) + (sel["clip_end"] - sel["clip_start"]) - inter_dur
                iou = inter_dur / max(0.01, union_dur)

                # Also suppress if they share the exact same gameplay trigger (within 4.0s)
                same_trigger = abs(c["gameplay_timestamp"] - sel["gameplay_timestamp"]) < 4.0

                if iou > iou_threshold or same_trigger:
                    keep = False
                    break

            if keep:
                selected.append(c)

        # Re-sort chronologically for downstream video production
        return sorted(selected, key=lambda c: c["clip_start"])


def np_median(lst: list[float]) -> float:
    """Lightweight median without heavy imports."""
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    return (s[mid] if n % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0)

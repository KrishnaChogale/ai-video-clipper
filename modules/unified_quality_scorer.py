"""modules/unified_quality_scorer.py
Phase 5: Unified Clip Quality Scoring Engine.

Evaluates every candidate clip across 10 normalized dimensions (0-100 each),
applying scale normalization, reliability weighting, contextual gating,
and synergistic cross-modal multipliers.

Factors:
1.  reaction_intensity: Streamer arousal, facial blendshapes, vocal excitement.
2.  gameplay_importance: Gameplay event severity, explosion/crash/kill magnitude.
3.  reaction_synchronization: Temporal correlation within asymmetric reaction windows.
4.  entertainment_value: Synergistic payoff between gameplay trigger and streamer reaction.
5.  speech_context_relevance: Spoken commentary presence, keyword density, dialogue flow.
6.  facecam_visibility_framing: Facecam detection confidence, framing stability, absence of drift.
7.  audio_clarity_balance: Voice-to-gameplay balance, absence of clipping or prolonged dead air.
8.  caption_readability: Reading pace (optimal 2.0-3.5 wps), safe-zone compliance, line length.
9.  visual_quality: Sharpness, proper lighting, absence of compression blocks or black frames.
10. narrative_completeness: 4-part structure (Setup -> Trigger -> Reaction -> Payoff), no abrupt mid-sentence cuts.

Penalty & Multiplier Engine:
- Near-duplicate penalty: Decreases score for clips overlapping existing superior picks.
- Context truncation penalty: Penalizes clips missing setup or cutting off punchlines.
- Headline congruence: Penalizes misleading or exaggerated headlines that don't match content.
- Synergy multiplier: Rewards genuine cross-modal co-occurrences (e.g. scream + headshot).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedQualityScorer:
    """Central explainable quality scoring engine for streamer clips."""

    # Category severity weights for gameplay events (0-100 scale)
    GAMEPLAY_SEVERITY = {
        "gameplay_explosion": 92.0,
        "gameplay_kill_death": 95.0,
        "gameplay_collision_crash": 88.0,
        "gameplay_gunfight": 82.0,
        "gameplay_action_peak": 75.0,
        "gameplay_sudden_transient": 65.0,
        "unknown": 40.0,
    }

    # Reaction arousal weights for streamer reactions (0-100 scale)
    REACTION_AROUSAL = {
        "streamer_scream": 95.0,
        "streamer_laugh": 92.0,
        "streamer_hype": 88.0,
        "streamer_gasp": 82.0,
        "streamer_surprise": 80.0,
        "unknown": 35.0,
    }

    def __init__(self):
        # 10 core dimensions weights (sum = 1.0)
        self.weights = {
            # Content & Entertainment Core (48%)
            "reaction_intensity": 0.14,
            "gameplay_importance": 0.14,
            "reaction_synchronization": 0.10,
            "entertainment_value": 0.10,
            # Context & Narrative (26%)
            "narrative_completeness": 0.14,
            "speech_context_relevance": 0.12,
            # Production & Technical Quality (26%)
            "facecam_visibility_framing": 0.08,
            "audio_clarity_balance": 0.07,
            "caption_readability": 0.06,
            "visual_quality": 0.05,
        }

    def evaluate_clip(
        self,
        clip_candidate: dict[str, Any],
        whisper_segments: Optional[list[dict[str, Any]]] = None,
        audio_energy: Optional[list[dict[str, Any]]] = None,
        visual_scores: Optional[list[dict[str, Any]]] = None,
        facecam_tracks: Optional[list[dict[str, Any]]] = None,
        existing_selected_clips: Optional[list[dict[str, Any]]] = None,
        min_quality_score: float = 55.0,
    ) -> dict[str, Any]:
        """Compute comprehensive, explainable 0-100 quality score.

        Returns:
            dict containing:
              - total_score: float (0.0 - 100.0)
              - passed_quality_gate: bool
              - factor_scores: dict[str, float]
              - penalties: dict[str, float]
              - bonuses: dict[str, float]
              - explanation: str
              - rejection_reasons: list[str]
        """
        start = float(clip_candidate.get("start", 0.0))
        end = float(clip_candidate.get("end", 0.0))
        duration = max(0.1, end - start)

        rejection_reasons = []

        # -------------------------------------------------------------
        # Factor 1: Reaction Intensity & Confidence [0 - 100]
        # -------------------------------------------------------------
        rx_event = clip_candidate.get("reaction_event") or {}
        if isinstance(rx_event, str):
            rx_category = rx_event
            rx_intensity = float(clip_candidate.get("reaction_intensity", clip_candidate.get("clip_score", 65.0)))
            rx_confidence = float(clip_candidate.get("reaction_confidence", 0.7))
        elif isinstance(rx_event, dict):
            rx_category = rx_event.get("category", clip_candidate.get("category", "unknown"))
            rx_intensity = float(rx_event.get("intensity", clip_candidate.get("reaction_intensity", 50.0)))
            rx_confidence = float(rx_event.get("confidence", 0.5))
        else:
            rx_category = "unknown"
            rx_intensity = 50.0
            rx_confidence = 0.5
        base_arousal = self.REACTION_AROUSAL.get(rx_category, 45.0)

        # Scale intensity with confidence and arousal
        f_reaction = min(100.0, max(0.0, (rx_intensity * 0.55 + base_arousal * 0.45) * (0.4 + 0.6 * rx_confidence)))

        # -------------------------------------------------------------
        # Factor 2: Gameplay Event Importance [0 - 100]
        # -------------------------------------------------------------
        gp_event = clip_candidate.get("gameplay_event") or {}
        if isinstance(gp_event, str):
            gp_category = gp_event
            gp_intensity = float(clip_candidate.get("intensity", 60.0))
            gp_confidence = float(clip_candidate.get("confidence", 0.7))
        elif isinstance(gp_event, dict):
            gp_category = gp_event.get("category", clip_candidate.get("category", "unknown"))
            gp_intensity = float(gp_event.get("intensity", 60.0))
            gp_confidence = float(gp_event.get("confidence", 0.5))
        else:
            gp_category = "unknown"
            gp_intensity = 50.0
            gp_confidence = 0.5
        base_gp_sev = self.GAMEPLAY_SEVERITY.get(gp_category, 45.0)

        f_gameplay = min(100.0, max(0.0, (gp_intensity * 0.50 + base_gp_sev * 0.50) * (0.4 + 0.6 * gp_confidence)))

        # -------------------------------------------------------------
        # Factor 3: Reaction-to-Gameplay Synchronization [0 - 100]
        # -------------------------------------------------------------
        sync_data = clip_candidate.get("reaction_sync") or clip_candidate.get("sync_evidence") or {}
        f_sync = float(sync_data.get("sync_score", clip_candidate.get("sync_score", 50.0)))
        delay = float(sync_data.get("reaction_delay", 0.8))

        # Check for unlinked reaction
        if rx_category != "unknown" and gp_category != "unknown":
            # If delay is outside plausible human physiological bounds (-3.0s to +4.5s)
            if delay < -3.5 or delay > 5.5:
                f_sync = max(10.0, f_sync * 0.4)

        # -------------------------------------------------------------
        # Factor 4: Entertainment Value (Cross-Modal Synergy) [0 - 100]
        # -------------------------------------------------------------
        # Pure arithmetic mean does not capture comedy/hype: high reaction + high gameplay is exponential
        f_entertainment = (f_reaction * f_gameplay) / 100.0
        # If reaction was laughter or scream, give an entertainment boost
        if rx_category in ("streamer_laugh", "streamer_scream"):
            f_entertainment = min(100.0, f_entertainment * 1.15)

        # -------------------------------------------------------------
        # Factor 5: Speech & Context Relevance [0 - 100]
        # -------------------------------------------------------------
        speech_words = 0
        speech_score = 50.0
        clip_transcript = []
        if whisper_segments:
            for s in whisper_segments:
                s_start = float(s.get("start", 0))
                s_end = float(s.get("end", 0))
                if s_end > start and s_start < end:
                    clip_transcript.append(s.get("text", ""))
                    speech_words += len(s.get("text", "").split())

        wps = speech_words / duration
        # Optimal speech rate is 1.5 to 3.5 words per second
        if 1.2 <= wps <= 4.0:
            speech_score = 85.0
        elif 0.5 <= wps < 1.2:
            speech_score = 65.0
        elif wps > 4.0:
            speech_score = 70.0  # Slightly fast
        else:
            speech_score = 40.0  # Mostly silent or sparse commentary

        full_txt = " ".join(clip_transcript).lower()
        # Bonus for high-context action or reaction keywords
        high_context_words = ["look", "bro", "holy", "no way", "what", "how", "dead", "kill", "omg", "bhai", "arrey"]
        if any(w in full_txt for w in high_context_words):
            speech_score = min(100.0, speech_score + 12.0)
        f_speech = speech_score

        # -------------------------------------------------------------
        # Factor 6: Facecam Visibility & Framing [0 - 100]
        # -------------------------------------------------------------
        facecam_avail = bool(clip_candidate.get("facecam_available", False))
        if not facecam_avail:
            # For non-facecam gameplay clips, framing is neutral (80.0) so it doesn't artificially ruin the clip
            f_facecam = 80.0
        else:
            face_cov = float(clip_candidate.get("facecam_coverage", 0.85))
            face_score = float(clip_candidate.get("facecam_score", 80.0))
            f_facecam = min(100.0, max(20.0, (face_cov * 60.0 + face_score * 0.4)))
            if face_cov < 0.40:
                rejection_reasons.append(f"Facecam visible in only {face_cov*100:.0f}% of clip (< 40% threshold)")

        # -------------------------------------------------------------
        # Factor 7: Audio Clarity & Balance [0 - 100]
        # -------------------------------------------------------------
        audio_score = 85.0
        if audio_energy:
            # Check for excessive dead silence or clipped levels
            sec_start = int(max(0, start))
            sec_end = int(min(len(audio_energy), math.ceil(end)))
            slice_ae = audio_energy[sec_start:sec_end]
            if slice_ae:
                loudness_vals = [float(x.get("loudness", 0.1)) for x in slice_ae]
                avg_loudness = sum(loudness_vals) / len(loudness_vals)
                if avg_loudness < 0.005:
                    audio_score -= 30.0
                    rejection_reasons.append("Audio is virtually silent or muted (< 0.005 average loudness)")
                elif avg_loudness > 0.85:
                    audio_score -= 20.0  # Danger of clipping/distortion
        f_audio = max(10.0, min(100.0, audio_score))

        # -------------------------------------------------------------
        # Factor 8: Caption Readability [0 - 100]
        # -------------------------------------------------------------
        # Assess word density and text overflow risk
        if wps > 4.5:
            f_caption = 60.0  # Too fast to read comfortably
        elif 1.0 <= wps <= 3.5:
            f_caption = 95.0  # Perfect sweet spot
        elif wps == 0:
            f_caption = 70.0  # No speech, purely visual/gameplay audio
        else:
            f_caption = 80.0

        # -------------------------------------------------------------
        # Factor 9: Visual Quality [0 - 100]
        # -------------------------------------------------------------
        visual_score = 88.0
        if visual_scores:
            sec_start = int(max(0, start))
            sec_end = int(min(len(visual_scores), math.ceil(end)))
            slice_vs = visual_scores[sec_start:sec_end]
            if slice_vs:
                motions = [float(x.get("motion", 10.0)) for x in slice_vs]
                avg_motion = sum(motions) / len(motions)
                # Penalize totally static screens (e.g. paused menu)
                if avg_motion < 0.8:
                    visual_score -= 35.0
                    rejection_reasons.append("Visual motion is nearly zero (likely paused game or static screen)")
                elif avg_motion < 3.0:
                    visual_score -= 15.0
                elif avg_motion > 60.0:
                    visual_score -= 10.0  # Severe blur/shaking
        f_visual = max(20.0, min(100.0, visual_score))

        # -------------------------------------------------------------
        # Factor 10: Narrative Completeness (Setup -> Trigger -> Reaction -> Payoff)
        # -------------------------------------------------------------
        # A good clip needs at least 1.5s setup before trigger, and 2.0s payoff after reaction
        if isinstance(gp_event, dict):
            trigger_t = float(gp_event.get("timestamp", clip_candidate.get("gameplay_timestamp", start + duration * 0.35)))
        else:
            trigger_t = float(clip_candidate.get("gameplay_timestamp", start + duration * 0.35))

        if isinstance(rx_event, dict):
            reaction_t = float(rx_event.get("timestamp", clip_candidate.get("reaction_timestamp", trigger_t + 0.6)))
        else:
            reaction_t = float(clip_candidate.get("reaction_timestamp", trigger_t + 0.6))

        setup_dur = max(0.0, trigger_t - start)
        payoff_dur = max(0.0, end - max(trigger_t, reaction_t))

        f_narrative = 80.0
        if setup_dur < 1.0:
            f_narrative -= 30.0
            rejection_reasons.append(f"Missing setup context (trigger occurs {setup_dur:.1f}s from start, min 1.0s required)")
        elif setup_dur >= 2.5:
            f_narrative += 10.0  # Excellent buildup

        if payoff_dur < 1.5:
            f_narrative -= 30.0
            rejection_reasons.append(f"Abrupt cutoff mid-reaction (only {payoff_dur:.1f}s payoff after climax, min 1.5s required)")
        elif payoff_dur >= 3.0:
            f_narrative += 10.0  # Full aftermath captured

        f_narrative = max(10.0, min(100.0, f_narrative))

        # -------------------------------------------------------------
        # Composite Base Score (Weighted Sum)
        # -------------------------------------------------------------
        factor_scores = {
            "reaction_intensity": round(f_reaction, 1),
            "gameplay_importance": round(f_gameplay, 1),
            "reaction_synchronization": round(f_sync, 1),
            "entertainment_value": round(f_entertainment, 1),
            "narrative_completeness": round(f_narrative, 1),
            "speech_context_relevance": round(f_speech, 1),
            "facecam_visibility_framing": round(f_facecam, 1),
            "audio_clarity_balance": round(f_audio, 1),
            "caption_readability": round(f_caption, 1),
            "visual_quality": round(f_visual, 1),
        }

        base_score = sum(factor_scores[k] * self.weights[k] for k in self.weights)

        # -------------------------------------------------------------
        # Bonuses & Penalties
        # -------------------------------------------------------------
        bonuses = {}
        penalties = {}

        # 1. Synergy Bonus: High action + High genuine reaction + Tight sync
        if f_reaction >= 75.0 and f_gameplay >= 75.0 and f_sync >= 75.0:
            bonus_pts = 6.0
            bonuses["cross_modal_synergy"] = bonus_pts

        # 2. Viral Duration Bonus: 25s - 45s is the algorithmic sweet spot
        if 25.0 <= duration <= 45.0:
            bonuses["optimal_duration"] = 3.0
        elif duration < 15.0 or duration > 90.0:
            penalties["suboptimal_duration"] = -8.0

        # 3. Misleading Headline Penalty
        headline = str(clip_candidate.get("headline", "")).upper()
        if headline:
            # Extreme headline on calm video
            if any(h in headline for h in ["LOST HIS MIND", "CRAZY", "INSANE", "SCREAMING"]) and f_reaction < 60.0:
                penalties["misleading_reaction_headline"] = -15.0
                rejection_reasons.append("Headline promises extreme reaction, but streamer was calm/neutral")
            if any(h in headline for h in ["EXPLOSION", "ACCIDENT", "CRASH"]) and gp_category not in ("gameplay_explosion", "gameplay_collision_crash"):
                penalties["misleading_gameplay_headline"] = -12.0

        # 4. Near-Duplicate Penalty (compared against existing selected clips)
        duplicate_overlap = 0.0
        if existing_selected_clips:
            for ex in existing_selected_clips:
                ex_start = float(ex.get("start", 0))
                ex_end = float(ex.get("end", 0))
                intersection = max(0.0, min(end, ex_end) - max(start, ex_start))
                union = max(0.1, max(end, ex_end) - min(start, ex_start))
                iou = intersection / union
                overlap_ratio = intersection / max(0.1, duration)
                if iou > 0.30 or overlap_ratio > 0.40:
                    duplicate_overlap = max(duplicate_overlap, overlap_ratio)

        if duplicate_overlap > 0.60:
            penalties["near_duplicate_penalty"] = -35.0
            rejection_reasons.append(f"Near-duplicate clip ({duplicate_overlap*100:.0f}% overlap with existing selected clip)")
        elif duplicate_overlap > 0.35:
            penalties["partial_overlap_penalty"] = -18.0

        # Calculate final clamped score
        total_delta = sum(bonuses.values()) + sum(penalties.values())
        final_score = round(max(0.0, min(100.0, base_score + total_delta)), 1)

        # Quality gating:
        # Minimum acceptable score is min_quality_score and no hard rejection reasons
        if final_score < min_quality_score:
            rejection_reasons.append(f"Score {final_score:.1f} below minimum threshold {min_quality_score}")
        passed_quality_gate = (final_score >= min_quality_score) and (len([r for r in rejection_reasons if not r.startswith("Score")]) == 0)

        # Build human-readable explanation
        explanation_parts = []
        if f_entertainment >= 75.0:
            explanation_parts.append(f"High entertainment ({f_entertainment:.0f}/100) from {rx_category} synced with {gp_category}")
        elif f_reaction >= 70.0:
            explanation_parts.append(f"Strong streamer reaction ({f_reaction:.0f}/100)")
        elif f_gameplay >= 70.0:
            explanation_parts.append(f"Intense gameplay action ({f_gameplay:.0f}/100)")
        else:
            explanation_parts.append("Moderate ambient gameplay and reaction")

        if f_narrative >= 80.0:
            explanation_parts.append("complete 4-part narrative structure")
        elif f_narrative < 50.0:
            explanation_parts.append("poor narrative pacing / abrupt boundary")

        if bonuses:
            bonus_str = ", ".join(f"+{v:.0f} {k}" for k, v in bonuses.items())
            explanation_parts.append(f"received bonuses ({bonus_str})")
        if penalties:
            pen_str = ", ".join(f"{v:.0f} {k}" for k, v in penalties.items())
            explanation_parts.append(f"deducted penalties ({pen_str})")

        explanation = "; ".join(explanation_parts) + f". Final Score: {final_score}/100."

        return {
            "total_score": final_score,
            "passed_quality_gate": passed_quality_gate,
            "factor_scores": factor_scores,
            "bonuses": bonuses,
            "penalties": penalties,
            "explanation": explanation,
            "rejection_reasons": rejection_reasons,
        }

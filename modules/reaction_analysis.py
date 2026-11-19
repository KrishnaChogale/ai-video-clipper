"""Multimodal Vocal Arousal & Facial Reaction Analysis Engine (Phase 2).

Analyzes MediaPipe facial blendshapes, F0 pitch tracking, spectral centroid,
spectral flux, vocal arousal, and speech context to detect and classify streamer reactions:
- streamer_laugh
- streamer_scream
- streamer_gasp
- streamer_hype
- streamer_surprise
- streamer_anger
- streamer_confusion
- streamer_disappointment
- unknown (when evidence is insufficient)

Operates 100% on CPU (0 MB GPU VRAM) preserving RTX 3050 headroom for other modules.
"""
from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

REACTION_CATEGORIES = [
    "streamer_laugh",
    "streamer_scream",
    "streamer_gasp",
    "streamer_hype",
    "streamer_surprise",
    "streamer_anger",
    "streamer_confusion",
    "streamer_disappointment",
    "unknown",
]


class FacialReactionAnalyzer:
    """Analyze temporal sequence of MediaPipe FACS blendshapes with EMA smoothing."""

    def __init__(self, alpha: float = 0.30):
        self.alpha = alpha
        self.smoothed_blendshapes: dict[str, float] = {}

    def reset(self):
        self.smoothed_blendshapes = {}

    def update(self, raw_blendshapes: dict[str, float]) -> dict[str, float]:
        """Apply rolling exponential moving average smoothing across blendshapes."""
        if not raw_blendshapes:
            # Decay towards zero
            for k in list(self.smoothed_blendshapes.keys()):
                self.smoothed_blendshapes[k] = round(self.smoothed_blendshapes[k] * 0.7, 4)
            return dict(self.smoothed_blendshapes)

        smoothed = {}
        for k, v in raw_blendshapes.items():
            prev = self.smoothed_blendshapes.get(k, v)
            sm = self.alpha * v + (1.0 - self.alpha) * prev
            smoothed[k] = round(sm, 4)
        self.smoothed_blendshapes = smoothed
        return dict(smoothed)

    def extract_cues(self, blendshapes: dict[str, float], face_motion: float = 0.0) -> dict[str, float]:
        """Extract evidence scores (0.0 to 1.0) for individual facial reaction expressions."""
        if not blendshapes:
            return {
                "laugh": 0.0,
                "scream": 0.0,
                "gasp": 0.0,
                "surprise": 0.0,
                "anger": 0.0,
                "confusion": 0.0,
                "disappointment": 0.0,
                "intensity": 0.0,
            }

        smile_l = blendshapes.get("mouthSmileLeft", 0.0)
        smile_r = blendshapes.get("mouthSmileRight", 0.0)
        smile = max(smile_l, smile_r)

        jaw_open = blendshapes.get("jawOpen", 0.0)
        brow_inner_up = blendshapes.get("browInnerUp", 0.0)
        brow_down_l = blendshapes.get("browDownLeft", 0.0)
        brow_down_r = blendshapes.get("browDownRight", 0.0)
        brow_down = max(brow_down_l, brow_down_r)

        eye_wide_l = blendshapes.get("eyeWideLeft", 0.0)
        eye_wide_r = blendshapes.get("eyeWideRight", 0.0)
        eye_wide = max(eye_wide_l, eye_wide_r)

        frown_l = blendshapes.get("mouthFrownLeft", 0.0)
        frown_r = blendshapes.get("mouthFrownRight", 0.0)
        frown = max(frown_l, frown_r)

        mouth_pucker = blendshapes.get("mouthPucker", 0.0)
        mouth_funnel = blendshapes.get("mouthFunnel", 0.0)
        o_mouth = max(mouth_pucker, mouth_funnel)

        # 1. Laugh: Smile + jaw open + motion
        laugh_score = min(1.0, (smile * 1.4 + jaw_open * 0.6) * (1.0 if smile > 0.35 else 0.4))

        # 2. Scream: Wide open jaw + wide eyes + high motion
        scream_score = min(1.0, (jaw_open * 1.2 + eye_wide * 0.8) * (1.0 if jaw_open > 0.45 else 0.3))

        # 3. Gasp: Wide eyes + brow raised + moderate jaw open
        gasp_score = min(1.0, (eye_wide * 1.0 + brow_inner_up * 0.7 + jaw_open * 0.5) * (1.0 if eye_wide > 0.30 else 0.3))

        # 4. Surprise: Brow inner up + eye wide + O-mouth
        surprise_score = min(1.0, (brow_inner_up * 0.9 + eye_wide * 0.6 + o_mouth * 0.5))

        # 5. Anger: Brow down (furrow) + jaw clench or shout
        anger_score = min(1.0, (brow_down * 1.2 + frown * 0.5) * (1.0 if brow_down > 0.30 else 0.2))

        # 6. Confusion: Brow asymmetry between left/right OR simultaneous furrow + raise
        asym_lr = max(abs(brow_down_l - brow_down_r), abs(eye_wide_l - eye_wide_r))
        confused_brows = min(brow_inner_up, brow_down) * 2.0
        confusion_score = min(1.0, asym_lr * 1.2 + confused_brows + frown * 0.3)

        # 7. Disappointment: Frown + lack of smile + head drop
        disappointment_score = min(1.0, (frown * 1.5 - smile * 0.8)) if frown > 0.20 else 0.0

        # Overall facial reaction intensity (0.0 to 100.0)
        intensity = min(100.0, (
            max(laugh_score, scream_score, gasp_score, surprise_score, anger_score) * 65.0
            + min(35.0, face_motion * 0.7)
        ))

        return {
            "laugh": round(max(0.0, laugh_score), 3),
            "scream": round(max(0.0, scream_score), 3),
            "gasp": round(max(0.0, gasp_score), 3),
            "surprise": round(max(0.0, surprise_score), 3),
            "anger": round(max(0.0, anger_score), 3),
            "confusion": round(max(0.0, confusion_score), 3),
            "disappointment": round(max(0.0, disappointment_score), 3),
            "intensity": round(intensity, 1),
        }


class AudioReactionAnalyzer:
    """Analyze vocal arousal, pitch contours, and spectral characteristics to isolate voice cues."""

    def extract_cues(self, audio_bucket: dict[str, Any], has_speech_word: bool = False) -> dict[str, float]:
        """Extract vocal reaction evidence scores from audio features."""
        if not audio_bucket:
            return {
                "laugh": 0.0, "scream": 0.0, "gasp": 0.0, "hype": 0.0,
                "surprise": 0.0, "vocal_arousal": 0.0, "is_game_sfx": False
            }

        loudness = float(audio_bucket.get("loudness", 0.0))
        spike = float(audio_bucket.get("spike", 0.0))
        f0 = float(audio_bucket.get("f0", 0.0))
        f0_max = float(audio_bucket.get("f0_max", 0.0))
        centroid = float(audio_bucket.get("spectral_centroid", 0.0))
        flatness = float(audio_bucket.get("spectral_flatness", 0.0))
        arousal = float(audio_bucket.get("vocal_arousal", 0.0))
        voice_conf = float(audio_bucket.get("voice_confidence", 0.0))

        # Detect game SFX vs human voice:
        # Explosions and gunshots have high flatness (>0.25) and no voiced F0 harmonicity
        is_explosion = bool(audio_bucket.get("is_explosion", False))
        is_gunshot = bool(audio_bucket.get("is_gunshot", False))
        is_game_sfx = (is_explosion or is_gunshot or (flatness > 0.35 and f0 == 0.0)) and not has_speech_word

        if is_game_sfx:
            return {
                "laugh": 0.0, "scream": 0.0, "gasp": 0.0, "hype": 0.0,
                "surprise": 0.0, "vocal_arousal": 0.0, "is_game_sfx": True
            }

        # 1. Vocal Scream: Very high pitch + elevated RMS + bright spectral centroid + harmonic + onset burst
        scream_score = 0.0
        if f0_max > 360.0 and loudness > 0.10 and voice_conf > 0.25 and spike > 8.0:
            pitch_boost = min(1.0, (f0_max - 340.0) / 240.0)
            energy_boost = min(1.0, loudness / 0.25)
            scream_score = 0.6 * pitch_boost + 0.4 * energy_boost

        # 2. Vocal Laugh: Repeated onset bursts (spike > 8) + moderate pitch + voiced
        laugh_score = 0.0
        if spike > 8.0 and voice_conf > 0.20 and not is_game_sfx:
            burst_factor = min(1.0, (spike - 6.0) / 14.0)
            laugh_score = min(1.0, burst_factor * 0.8 + (0.2 if 150 < f0 < 450 else 0.0))

        # 3. Vocal Gasp: Sharp transient onset with low/moderate sustained loudness
        gasp_score = 0.0
        if spike > 12.0 and loudness < 0.12 and centroid > 1800:
            gasp_score = min(1.0, (spike / 25.0) * (1.0 - loudness / 0.20))

        # 4. Vocal Hype: High vocal arousal + active speech / shouting
        hype_score = 0.0
        if arousal > 45.0 and (voice_conf > 0.30 or has_speech_word):
            hype_score = min(1.0, (arousal - 40.0) / 50.0)

        # 5. Vocal Surprise: Pitch jump / sudden stop
        surprise_score = 0.0
        if f0_max > 260.0 and spike > 10.0:
            surprise_score = min(1.0, ((f0_max - 240.0) / 200.0) * 0.6 + (spike / 25.0) * 0.4)

        return {
            "laugh": round(scream_score * 0.1 + laugh_score * 0.9, 3),
            "scream": round(scream_score, 3),
            "gasp": round(gasp_score, 3),
            "hype": round(hype_score, 3),
            "surprise": round(surprise_score, 3),
            "vocal_arousal": round(arousal, 1),
            "is_game_sfx": False,
        }


class MultimodalReactionClassifier:
    """Fuse facial landmarks, vocal arousal, and speech context to detect reaction events."""

    def __init__(self, confidence_threshold: float = 0.38):
        self.confidence_threshold = confidence_threshold
        self.facial_analyzer = FacialReactionAnalyzer()
        self.audio_analyzer = AudioReactionAnalyzer()

    def classify_second(
        self,
        visual_item: dict[str, Any] | None,
        audio_item: dict[str, Any] | None,
        speech_text: str = "",
    ) -> dict[str, Any]:
        """Classify a single second of multimodal data into a reaction category with evidence."""
        has_face = bool(visual_item and visual_item.get("facecam_box"))
        raw_blendshapes = visual_item.get("blendshapes", {}) if visual_item else {}
        face_motion = float(visual_item.get("facecam_motion", 0.0)) if visual_item else 0.0
        has_speech = bool(speech_text and len(speech_text.strip()) > 0)

        # 1. Facial Cues
        smooth_shapes = self.facial_analyzer.update(raw_blendshapes)
        f_cues = self.facial_analyzer.extract_cues(smooth_shapes, face_motion) if has_face else {}

        # 2. Audio Cues
        a_cues = self.audio_analyzer.extract_cues(audio_item or {}, has_speech_word=has_speech)

        # Check speech keywords for contextual reinforcement
        text_lower = speech_text.lower()
        laugh_word = any(w in text_lower for w in ["haha", "lmao", "lol", "hehe", "rofl"])
        hype_word = any(w in text_lower for w in ["let's go", "lets go", "no way", "oh my god", "omg", "holy", "insane"])
        confused_word = any(w in text_lower for w in ["what", "wait", "huh", "how", "why"])

        # 3. Multimodal Evidence Synthesis
        category_scores: dict[str, float] = {cat: 0.0 for cat in REACTION_CATEGORIES if cat != "unknown"}

        if has_face:
            # Multimodal Mode: Face + Audio
            # Enforce physical face validation: if the streamer's face is visibly still and neutral,
            # external audio (teammates, game sound) should not trigger streamer reaction labels.
            f_laugh = f_cues.get("laugh", 0.0)
            f_scream = f_cues.get("scream", 0.0)
            f_gasp = f_cues.get("gasp", 0.0)
            f_intensity = f_cues.get("intensity", 0.0) / 100.0

            category_scores["streamer_laugh"] = (
                (0.55 * f_laugh + 0.40 * a_cues.get("laugh", 0.0) + (0.20 if laugh_word else 0.0))
                if (f_laugh >= 0.15 or (f_intensity >= 0.30 and laugh_word)) else 0.0
            )
            category_scores["streamer_scream"] = (
                (0.50 * f_scream + 0.50 * a_cues.get("scream", 0.0))
                if (f_scream >= 0.15 or (f_intensity >= 0.35 and a_cues.get("scream", 0.0) >= 0.40)) else 0.0
            )
            category_scores["streamer_gasp"] = (
                (0.55 * f_gasp + 0.45 * a_cues.get("gasp", 0.0))
                if f_gasp >= 0.20 else 0.0
            )
            category_scores["streamer_hype"] = (
                0.35 * f_intensity + 0.45 * a_cues.get("hype", 0.0) + (0.20 if hype_word else 0.0)
            )
            category_scores["streamer_surprise"] = (
                (0.60 * f_cues.get("surprise", 0.0) + 0.40 * a_cues.get("surprise", 0.0))
                if (f_cues.get("surprise", 0.0) >= 0.20 or confused_word) else 0.0
            )
            category_scores["streamer_anger"] = (
                (0.75 * f_cues.get("anger", 0.0) + 0.25 * (1.0 if a_cues.get("vocal_arousal", 0.0) > 40 else 0.0))
                if f_cues.get("anger", 0.0) >= 0.25 else 0.0
            )
            category_scores["streamer_confusion"] = (
                (0.70 * f_cues.get("confusion", 0.0) + (0.30 if confused_word else 0.0))
                if (f_cues.get("confusion", 0.0) >= 0.20 or confused_word) else 0.0
            )
            category_scores["streamer_disappointment"] = (
                (0.80 * f_cues.get("disappointment", 0.0) + 0.20 * (1.0 if a_cues.get("vocal_arousal", 0.0) < 20 else 0.0))
                if f_cues.get("disappointment", 0.0) >= 0.25 else 0.0
            )
        else:
            # Audio-Only Mode: When facecam is not present
            # Require higher acoustic confidence to prevent singing or background music from false-triggering
            a_scream = a_cues.get("scream", 0.0)
            a_laugh = a_cues.get("laugh", 0.0)
            a_hype = a_cues.get("hype", 0.0)
            vocal_arousal = float(a_cues.get("vocal_arousal", 0.0))

            category_scores["streamer_scream"] = 0.90 * a_scream if (a_scream > 0.45 and vocal_arousal > 60.0) else 0.0
            category_scores["streamer_laugh"] = (0.75 * a_laugh + (0.25 if laugh_word else 0.0)) if (a_laugh > 0.45 or laugh_word) else 0.0
            category_scores["streamer_gasp"] = 0.85 * a_cues.get("gasp", 0.0) if a_cues.get("gasp", 0.0) > 0.45 else 0.0
            category_scores["streamer_hype"] = 0.80 * a_hype + (0.20 if hype_word else 0.0)
            category_scores["streamer_surprise"] = 0.75 * a_cues.get("surprise", 0.0) if a_cues.get("surprise", 0.0) > 0.45 else 0.0
            category_scores["streamer_anger"] = 0.0
            category_scores["streamer_confusion"] = 0.40 if confused_word else 0.0
            category_scores["streamer_disappointment"] = 0.0

        # Find best category
        best_cat, best_conf = max(category_scores.items(), key=lambda item: item[1])

        # Reject if below confidence threshold or if it was pure game SFX
        if a_cues.get("is_game_sfx", False) or best_conf < self.confidence_threshold:
            best_cat = "unknown"
            final_conf = 0.0
        else:
            final_conf = round(min(1.0, best_conf), 3)

        # Intensity (0.0 to 100.0)
        total_intensity = min(100.0, max(
            f_cues.get("intensity", 0.0),
            a_cues.get("vocal_arousal", 0.0)
        ))

        return {
            "category": best_cat,
            "confidence": final_conf,
            "intensity": round(total_intensity, 1),
            "evidence": {
                "facial": f_cues,
                "audio": a_cues,
                "speech_text": speech_text.strip(),
            }
        }

    def analyze_timeline(
        self,
        visual_scores: list[dict[str, Any]],
        audio_buckets: list[dict[str, Any]],
        whisper_segments: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Analyze full timeline and return per-second classifications and aggregated reaction events.
        
        Returns:
            (per_second_results, reaction_events)
        """
        total_seconds = max(len(visual_scores), len(audio_buckets))
        per_second = []

        # Index speech words by second
        speech_by_second: dict[int, list[str]] = {}
        if whisper_segments:
            for seg in whisper_segments:
                start_s = int(seg.get("start", 0))
                end_s = int(seg.get("end", 0))
                text = seg.get("text", "")
                for s in range(start_s, end_s + 1):
                    speech_by_second.setdefault(s, []).append(text)

        self.facial_analyzer.reset()

        for idx in range(total_seconds):
            v_item = visual_scores[idx] if idx < len(visual_scores) else None
            a_item = audio_buckets[idx] if idx < len(audio_buckets) else None

            # Retain input timestamp if provided
            actual_sec = idx
            if a_item is not None and "second" in a_item:
                actual_sec = a_item["second"]
            elif v_item is not None and "second" in v_item:
                actual_sec = v_item["second"]

            text = " ".join(speech_by_second.get(actual_sec, []))

            cls_res = self.classify_second(v_item, a_item, speech_text=text)
            cls_res["second"] = actual_sec
            per_second.append(cls_res)

        # Aggregate contiguous seconds into reaction events (min duration: 1.0s)
        events = []
        cur_event = None

        def finalize_event(ev: dict[str, Any] | None) -> dict[str, Any] | None:
            if ev is None:
                return None
            ev["duration"] = round(float(ev["end"]) - float(ev["start"]), 2)
            if ev["duration"] >= 1.0:
                ev["confidence"] = round(float(np.mean(ev["_confs"])), 3)
                ev["intensity"] = round(float(np.max(ev["_intensities"])), 1)
                ev.pop("_confs", None)
                ev.pop("_intensities", None)
                ev.pop("_last_sec", None)
                events.append(ev)
            return None

        for item in per_second:
            cat = item["category"]
            sec = float(item["second"])
            bucket_dur = float(item.get("duration", 1.0))

            if cat == "unknown":
                cur_event = finalize_event(cur_event)
                continue

            if cur_event is None:
                cur_event = {
                    "start": sec,
                    "end": sec + bucket_dur,
                    "category": cat,
                    "evidence": item["evidence"],
                    "_confs": [item["confidence"]],
                    "_intensities": [item["intensity"]],
                    "_last_sec": sec,
                }
            elif cur_event["category"] == cat and abs(sec - (cur_event["_last_sec"] + bucket_dur)) < 0.05:
                # Contiguous second with identical category: extend event boundary
                cur_event["end"] = sec + bucket_dur
                cur_event["_confs"].append(item["confidence"])
                cur_event["_intensities"].append(item["intensity"])
                cur_event["_last_sec"] = sec
            else:
                # Category changed or time discontinuity: finalize previous and begin new
                cur_event = finalize_event(cur_event)
                cur_event = {
                    "start": sec,
                    "end": sec + bucket_dur,
                    "category": cat,
                    "evidence": item["evidence"],
                    "_confs": [item["confidence"]],
                    "_intensities": [item["intensity"]],
                    "_last_sec": sec,
                }

        finalize_event(cur_event)
        return per_second, events

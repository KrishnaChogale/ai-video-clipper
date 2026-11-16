"""
modules/audio_analysis.py
Analyses audio energy, onset spikes, and gunshot/explosion detection.
Reuses the working librosa + decode_audio approach from the original app.py.
"""
import logging
import numpy as np
import librosa
from faster_whisper.audio import decode_audio

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def load_audio(video_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio as a 16kHz mono float32 numpy array.
    Falls back to decode_audio (via PyAV) if libsndfile cannot parse the container.
    """
    try:
        y, sr = librosa.load(video_path, sr=config.AUDIO_SAMPLE_RATE, mono=True)
        logger.debug("Audio loaded via librosa: %.1f s", len(y) / sr)
        return y, sr
    except Exception as e:
        logger.debug("librosa.load fallback to decode_audio (%s)", e)
        y = decode_audio(video_path, sampling_rate=config.AUDIO_SAMPLE_RATE)
        return y, config.AUDIO_SAMPLE_RATE


def compute_energy_buckets(y: np.ndarray, sr: int) -> list[dict]:
    """
    Returns one dict per second:
      { second, loudness, spike, is_gunshot, is_explosion }
    """
    hop = config.HOP_LENGTH
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    frames_per_sec = sr / hop
    duration_sec = int(librosa.get_duration(y=y, sr=sr))

    # Gunshot / explosion heuristic: very high onset strength spike
    rms_max = float(np.percentile(rms, 99)) if len(rms) > 0 else 1.0
    onset_max = float(np.percentile(onset_env, 99)) if len(onset_env) > 0 else 1.0

    buckets = []
    for sec in range(duration_sec):
        s = int(sec * frames_per_sec)
        e = int((sec + 1) * frames_per_sec)
        avg_rms  = float(np.mean(rms[s:e]))   if e > s else 0.0
        max_onset = float(np.max(onset_env[s:e])) if e > s else 0.0

        # Classify loud transients as gunshot or explosion
        is_gunshot   = (avg_rms > 0.6 * rms_max and max_onset > 0.7 * onset_max)
        is_explosion = (avg_rms > 0.8 * rms_max)

        buckets.append({
            "second":       sec,
            "loudness":     round(avg_rms, 4),
            "spike":        round(max_onset, 2),
            "is_gunshot":   is_gunshot,
            "is_explosion": is_explosion,
        })

    return buckets


def analyze_audio(video_path: str) -> tuple[list[dict], np.ndarray, int]:
    """
    Primary audio analysis: loads audio once, returns energy buckets and the numpy array.
    Returns: (energy_buckets, y, sr)
    """
    y, sr = load_audio(video_path)
    energy_buckets = compute_energy_buckets(y, sr)
    return energy_buckets, y, sr

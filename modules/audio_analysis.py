"""modules/audio_analysis.py
Analyses audio energy, F0 fundamental frequency, spectral centroid, spectral flux,
vocal arousal, and gunshot/explosion vs voice discrimination.

Reuses librosa + decode_audio with vectorised signal processing on CPU (0 MB VRAM).
Maintains 100% backward compatibility with existing energy bucket keys.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import numpy as np
import librosa
from faster_whisper.audio import decode_audio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


def get_audio_duration(video_path: str) -> float:
    """Fast probe of video/audio duration in seconds via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception as e:
        logger.debug("ffprobe duration probe failed: %s", e)

    # Fallback to cv2
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if frames > 0 and fps > 0:
            return float(frames / fps)
    except Exception:
        pass
    return 0.0


def decode_audio_chunk(video_path: str, start_sec: float, duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Decode a bounded slice of audio directly from video using FFmpeg.
    
    Avoids loading full multi-hour audio files into memory. Returns float32 mono array at sr Hz.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0.0, start_sec)),
        "-i", video_path,
        "-t", str(max(0.1, duration_sec)),
        "-af", "aresample=async=1000",
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-",
    ]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw, _ = p.communicate(timeout=120)
        if not raw:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.float32)
    except Exception as e:
        logger.warning("decode_audio_chunk failed for %s [%.1fs - %.1fs]: %s", video_path, start_sec, start_sec + duration_sec, e)
        return np.zeros(0, dtype=np.float32)


def load_audio(video_path: str) -> tuple[np.ndarray, int]:
    """Load audio as a 16kHz mono float32 numpy array.
    
    Falls back to decode_audio (via PyAV) if libsndfile cannot parse the container.
    If the video container has no audio stream, gracefully returns a silent float32 array.
    """
    sr = config.AUDIO_SAMPLE_RATE
    try:
        y, _ = librosa.load(video_path, sr=sr, mono=True)
        logger.debug("Audio loaded via librosa: %.1f s", len(y) / sr)
        return y, sr
    except Exception as e:
        logger.debug("librosa.load fallback to decode_audio_chunk (%s)", e)
        try:
            dur = get_audio_duration(video_path)
            if dur > 0:
                y = decode_audio_chunk(video_path, 0.0, dur, sr=sr)
                if len(y) > 0:
                    return y, sr
            y = decode_audio(video_path, sampling_rate=sr)
            return y, sr
        except Exception as e2:
            logger.warning("No audio stream or failed to decode audio from %s: %s", video_path, e2)
            duration_s = 1.0
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                if frames > 0 and fps > 0:
                    duration_s = frames / fps
            except Exception:
                pass
            silent = np.zeros(int(duration_s * sr), dtype=np.float32)
            return silent, sr


def compute_energy_buckets(y: np.ndarray, sr: int, start_offset_sec: int = 0) -> list[dict]:
    """Returns one dict per second with energy, pitch, spectral, and vocal arousal features:
      {
        second, loudness, spike, is_gunshot, is_explosion,
        f0, f0_max, spectral_centroid, spectral_flux, spectral_flatness,
        vocal_arousal, voice_confidence
      }
    start_offset_sec offsets the 'second' index for chunked streaming of long streams.
    """
    if y is None or len(y) == 0:
        return []

    hop = config.HOP_LENGTH
    # 1. Base Energy and Spectral Features
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]

    # 2. Fundamental Frequency (F0) tracking via YIN algorithm
    # Use hop_f0 = hop * 2 (1024) for ultra-fast, CPU-friendly pitch extraction (~15.6 fps)
    hop_f0 = hop * 2
    try:
        raw_f0 = librosa.yin(y, fmin=65, fmax=600, sr=sr, hop_length=hop_f0)
        # Clean NaNs
        clean_f0 = np.where(np.isnan(raw_f0), 0.0, raw_f0)
    except Exception as e:
        logger.debug("YIN F0 tracking fallback to zero: %s", e)
        clean_f0 = np.zeros(len(y) // hop_f0 + 1, dtype=np.float32)

    frames_per_sec = sr / hop
    frames_per_sec_f0 = sr / hop_f0
    duration_sec = int(round(len(y) / sr))

    # Reference normalizers
    rms_max = float(np.percentile(rms, 99)) if len(rms) > 0 else 1.0
    rms_p90 = max(0.001, float(np.percentile(rms, 90))) if len(rms) > 0 else 0.1
    onset_max = float(np.percentile(onset_env, 99)) if len(onset_env) > 0 else 1.0

    buckets = []
    for sec in range(duration_sec):
        # Slice frame ranges for this second
        s = int(sec * frames_per_sec)
        e = int((sec + 1) * frames_per_sec)
        s_f0 = int(sec * frames_per_sec_f0)
        e_f0 = int((sec + 1) * frames_per_sec_f0)

        avg_rms = float(np.mean(rms[s:e])) if e > s and len(rms) >= e else 0.0
        max_onset = float(np.max(onset_env[s:e])) if e > s and len(onset_env) >= e else 0.0
        mean_cent = float(np.mean(centroid[s:e])) if e > s and len(centroid) >= e else 0.0
        mean_flat = float(np.mean(flatness[s:e])) if e > s and len(flatness) >= e else 0.0

        # Pitch slice
        f0_slice = clean_f0[s_f0:e_f0] if e_f0 > s_f0 and len(clean_f0) >= e_f0 else np.array([])
        voiced_f0 = f0_slice[f0_slice > 0.0]
        mean_f0 = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        max_f0 = float(np.max(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # Transients / Game sound effects (explosions, gunshots)
        is_gunshot = (avg_rms > 0.6 * rms_max and max_onset > 0.7 * onset_max)
        is_explosion = (avg_rms > 0.8 * rms_max and mean_flat > 0.25)

        # Voice confidence: harmonic tonality vs broadband unvoiced noise
        if mean_f0 > 0.0 and mean_flat < 0.20:
            tonality = max(0.0, 1.0 - mean_flat * 5.0)
            voiced_ratio = len(voiced_f0) / max(1, len(f0_slice))
            voice_conf = min(1.0, 0.5 * tonality + 0.5 * voiced_ratio)
        else:
            voice_conf = 0.05 if avg_rms > 0.01 else 0.0

        # Vocal Arousal (0.0 to 100.0):
        # Combines elevated pitch, energy (RMS), and spectral brightness
        if voice_conf > 0.20:
            pitch_score = min(100.0, max(0.0, (max_f0 - 110.0) / 3.0)) if max_f0 > 0 else 0.0
            energy_score = min(100.0, (avg_rms / rms_p90) * 80.0)
            brightness_score = min(100.0, (mean_cent / 3200.0) * 100.0)
            arousal = 0.45 * energy_score + 0.35 * pitch_score + 0.20 * brightness_score
        else:
            # Unvoiced/noise: dramatically dampen vocal arousal to prevent explosions as screams
            arousal = min(25.0, (avg_rms / rms_p90) * 20.0)

        buckets.append({
            "second": sec + start_offset_sec,
            "loudness": round(avg_rms, 4),
            "spike": round(max_onset, 2),
            "is_gunshot": is_gunshot,
            "is_explosion": is_explosion,
            "f0": round(mean_f0, 1),
            "f0_max": round(max_f0, 1),
            "spectral_centroid": round(mean_cent, 1),
            "spectral_flux": round(max_onset, 2),
            "spectral_flatness": round(mean_flat, 4),
            "vocal_arousal": round(max(0.0, min(100.0, arousal)), 2),
            "voice_confidence": round(max(0.0, min(1.0, voice_conf)), 2),
        })

    return buckets


def analyze_audio_chunked(
    video_path: str,
    chunk_duration_sec: float = 900.0,
    progress_callback=None,
) -> list[dict]:
    """Memory-bounded audio analysis for long streams.
    
    Processes audio in consecutive 15-minute chunks (~115 MB RAM per chunk),
    concatenating energy buckets and immediately discarding raw float32 arrays.
    Peak memory remains <= 150 MB even for 10-12+ hour files.
    """
    total_dur = get_audio_duration(video_path)
    sr = config.AUDIO_SAMPLE_RATE

    if total_dur <= 0 or total_dur <= chunk_duration_sec:
        y, _ = load_audio(video_path)
        return compute_energy_buckets(y, sr)

    all_buckets = []
    num_chunks = int(np.ceil(total_dur / chunk_duration_sec))
    logger.info("Starting chunked audio analysis for %s (%.1fs across %d chunks)", os.path.basename(video_path), total_dur, num_chunks)

    for i in range(num_chunks):
        c_start = i * chunk_duration_sec
        c_dur = min(chunk_duration_sec, total_dur - c_start)
        if c_dur <= 0:
            break
        chunk_y = decode_audio_chunk(video_path, start_sec=c_start, duration_sec=c_dur, sr=sr)
        if len(chunk_y) > 0:
            offset = len(all_buckets)
            c_buckets = compute_energy_buckets(chunk_y, sr=sr, start_offset_sec=offset)
            all_buckets.extend(c_buckets)
        del chunk_y
        if progress_callback:
            progress_callback(i + 1, num_chunks, f"Audio energy chunk {i+1}/{num_chunks}")

    logger.info("Chunked audio analysis complete: %d seconds analyzed", len(all_buckets))
    return all_buckets


_AUDIO_CACHE: dict[tuple[str, float], tuple[list[dict], np.ndarray, int]] = {}


def clear_audio_cache():
    """Clear cached audio buffers from memory."""
    global _AUDIO_CACHE
    _AUDIO_CACHE.clear()


def analyze_audio(video_path: str, use_cache: bool = True, chunk_if_long: bool = True) -> tuple[list[dict], np.ndarray, int]:
    """Primary audio analysis: returns (energy_buckets, y, sr).
    
    For videos > 30 minutes, computes energy buckets via streaming chunks
    to avoid allocating multiple gigabytes of memory.
    """
    global _AUDIO_CACHE
    mtime = os.path.getmtime(video_path) if os.path.isfile(video_path) else 0.0
    cache_key = (os.path.abspath(video_path), mtime)
    if use_cache and cache_key in _AUDIO_CACHE:
        logger.debug("Returning cached audio features for %s", video_path)
        return _AUDIO_CACHE[cache_key]

    total_dur = get_audio_duration(video_path)
    sr = config.AUDIO_SAMPLE_RATE

    # If stream is longer than 30 minutes (1800s) and chunk_if_long is enabled,
    # use chunked calculation so we don't hold 2+ GB in memory
    if chunk_if_long and total_dur > 1800.0:
        logger.info("Long stream detected (%.1fs > 1800s); activating chunked audio analysis", total_dur)
        energy_buckets = analyze_audio_chunked(video_path, chunk_duration_sec=900.0)
        # For long streams, return None for y so downstream modules stream audio rather than holding 2.15GB
        res = (energy_buckets, None, sr)
    else:
        y, sr = load_audio(video_path)
        energy_buckets = compute_energy_buckets(y, sr)
        res = (energy_buckets, y, sr)

    if use_cache:
        # Keep at most 1 video audio buffer to strictly preserve 16GB RAM headroom
        if len(_AUDIO_CACHE) >= 1:
            _AUDIO_CACHE.clear()
        _AUDIO_CACHE[cache_key] = res

    return res

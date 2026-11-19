"""
modules/speech_recognition.py
Faster-Whisper wrapper with CUDA / CPU auto-fallback.
Optimised for RTX 3050 Laptop GPU (6GB VRAM) and CPU fallback.
"""
import logging
import numpy as np
from typing import Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.gpu_utils import release_gpu_memory

logger = logging.getLogger(__name__)

_model_cache = {}   # Avoid reloading model within the same session


def _get_model(model_size: str, device: str):
    key = (model_size, device)
    if key in _model_cache:
        return _model_cache[key]

    # Ensure CUDA DLLs from PyTorch are registered in Windows DLL search path
    if device == "cuda":
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(torch_lib)
                if torch_lib not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
        except Exception:
            pass

    from faster_whisper import WhisperModel
    compute = config.WHISPER_COMPUTE_TYPE_CUDA if device == "cuda" else config.WHISPER_COMPUTE_TYPE_CPU
    logger.info("Loading Whisper '%s' on %s (%s)", model_size, device, compute)
    m = WhisperModel(model_size, device=device, compute_type=compute)
    _model_cache[key] = m
    return m


def _transcribe_single_segment(
    model,
    audio_data: np.ndarray,
    beam_size: int,
    vad_filter: bool = True,
    time_offset: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Transcribe a single audio chunk and return offset segments and words."""
    if len(audio_data) == 0:
        return [], []

    segs, _ = model.transcribe(
        audio_data,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
    )

    transcript = []
    words_list = []
    for seg in segs:
        clean_text = seg.text.strip()
        if not clean_text:
            continue
        transcript.append({
            "start": round(time_offset + float(seg.start), 2),
            "end":   round(time_offset + float(seg.end), 2),
            "text":  clean_text,
        })
        if getattr(seg, "words", None):
            for w in seg.words:
                w_text = w.word.strip()
                if w_text:
                    words_list.append({
                        "start": round(time_offset + float(w.start), 2),
                        "end":   round(time_offset + float(w.end), 2),
                        "word":  w_text,
                    })
    return transcript, words_list


def transcribe_audio(
    audio_input: Union[np.ndarray, str],
    model_size: str = None,
    beam_size: int = None,
    chunk_duration_sec: float = 900.0,
    vad_filter: bool = True,
    progress_callback = None,
) -> tuple[list[dict], list[dict]]:
    """
    Transcribe speech from a 16 kHz mono float32 numpy array OR an audio/video file path.
    For long audio (> 15 minutes), processes in memory-bounded 15-minute chunks
    to prevent allocating gigabytes of RAM while maintaining exact timeline continuity.

    Returns:
        (transcript_segments, word_level_timestamps)
    Auto-falls back from CUDA to CPU if cuBLAS/cuDNN DLLs encounter issues.
    """
    model_size = model_size or config.WHISPER_MODEL_SIZE
    beam_size = beam_size or config.WHISPER_BEAM_SIZE
    sr = config.AUDIO_SAMPLE_RATE

    from modules.audio_analysis import decode_audio_chunk, get_audio_duration, load_audio

    # Try preferred device order (CUDA -> CPU)
    devices_to_try = ("cuda", "cpu") if config.PREFERRED_DEVICE in ("auto", "cuda") else ("cpu",)
    model = None
    for device in devices_to_try:
        try:
            model = _get_model(model_size, device)
            break
        except Exception as e:
            logger.warning("Whisper model load on %s failed (%s)", device, e)
            release_gpu_memory()
            _model_cache.pop((model_size, device), None)
            if device == "cpu":
                raise e

    transcript = []
    words_list = []

    # Path A: Input is a video / audio file path string
    if isinstance(audio_input, str):
        total_dur = get_audio_duration(audio_input)
        if total_dur <= 0 or total_dur <= chunk_duration_sec:
            audio_data, _ = load_audio(audio_input)
            transcript, words_list = _transcribe_single_segment(model, audio_data, beam_size, vad_filter, time_offset=0.0)
            del audio_data
        else:
            num_chunks = int(np.ceil(total_dur / chunk_duration_sec))
            logger.info("Transcribing %s across %d chunks (%.1fs total)", os.path.basename(audio_input), num_chunks, total_dur)
            for i in range(num_chunks):
                c_start = i * chunk_duration_sec
                c_dur = min(chunk_duration_sec, total_dur - c_start)
                if c_dur <= 0:
                    break
                if progress_callback:
                    progress_callback(i + 1, num_chunks, f"Transcribing audio chunk {i+1}/{num_chunks} ({int(c_start/60)}m - {int((c_start+c_dur)/60)}m)")
                chunk_y = decode_audio_chunk(audio_input, start_sec=c_start, duration_sec=c_dur, sr=sr)
                c_trans, c_words = _transcribe_single_segment(model, chunk_y, beam_size, vad_filter, time_offset=c_start)
                transcript.extend(c_trans)
                words_list.extend(c_words)
                del chunk_y

    # Path B: Input is a float32 numpy array
    else:
        audio_data = audio_input
        total_samples = len(audio_data)
        chunk_samples = int(chunk_duration_sec * sr)
        if total_samples <= chunk_samples:
            transcript, words_list = _transcribe_single_segment(model, audio_data, beam_size, vad_filter, time_offset=0.0)
        else:
            num_chunks = int(np.ceil(total_samples / chunk_samples))
            logger.info("Transcribing audio array across %d chunks", num_chunks)
            for i in range(num_chunks):
                s_idx = i * chunk_samples
                e_idx = min(total_samples, s_idx + chunk_samples)
                c_start_sec = s_idx / sr
                if progress_callback:
                    progress_callback(i + 1, num_chunks, f"Transcribing audio chunk {i+1}/{num_chunks}")
                chunk_y = audio_data[s_idx:e_idx]
                c_trans, c_words = _transcribe_single_segment(model, chunk_y, beam_size, vad_filter, time_offset=c_start_sec)
                transcript.extend(c_trans)
                words_list.extend(c_words)

    logger.info("Transcribed %d segments, %d words", len(transcript), len(words_list))
    return transcript, words_list


def clear_model_cache():
    """Free Whisper models from memory."""
    _model_cache.clear()
    release_gpu_memory()

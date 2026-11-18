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


def transcribe_audio(
    audio_input: Union[np.ndarray, str],
    model_size: str = None,
    beam_size: int = None,
) -> tuple[list[dict], list[dict]]:
    """
    Transcribe speech from a 16 kHz mono float32 numpy array OR an audio/video file path.
    Returns:
        (transcript_segments, word_level_timestamps)
    Auto-falls back from CUDA to CPU if cuBLAS/cuDNN DLLs encounter issues.
    """
    model_size = model_size or config.WHISPER_MODEL_SIZE
    beam_size = beam_size or config.WHISPER_BEAM_SIZE

    # If audio_input is a path string, load it via decode_audio or librosa
    if isinstance(audio_input, str):
        from modules.audio_analysis import load_audio
        audio_data, _ = load_audio(audio_input)
    else:
        audio_data = audio_input

    seg_list = []
    # Try preferred device order (CUDA -> CPU)
    devices_to_try = ("cuda", "cpu") if config.PREFERRED_DEVICE in ("auto", "cuda") else ("cpu",)

    for device in devices_to_try:
        try:
            model = _get_model(model_size, device)
            segs, _ = model.transcribe(
                audio_data,
                beam_size=beam_size,
                word_timestamps=True,
            )
            seg_list = list(segs)
            break
        except Exception as e:
            logger.warning("Whisper transcription on %s failed (%s)", device, e)
            release_gpu_memory()
            _model_cache.pop((model_size, device), None)
            if device == "cpu":
                # Both failed or CPU failed
                raise e

    transcript = []
    words_list = []
    for seg in seg_list:
        clean_text = seg.text.strip()
        if not clean_text:
            continue
        transcript.append({
            "start": round(float(seg.start), 2),
            "end":   round(float(seg.end), 2),
            "text":  clean_text,
        })
        if getattr(seg, "words", None):
            for w in seg.words:
                w_text = w.word.strip()
                if w_text:
                    words_list.append({
                        "start": round(float(w.start), 2),
                        "end":   round(float(w.end), 2),
                        "word":  w_text,
                    })

    logger.info("Transcribed %d segments, %d words", len(transcript), len(words_list))
    return transcript, words_list


def clear_model_cache():
    """Free Whisper models from memory."""
    _model_cache.clear()
    release_gpu_memory()

"""
modules/object_detection.py
YOLO-based performer/person scoring with general scene activity weights.
Includes frame-difference motion detection and scene-change cuts.
Optimised for fast frame sampling via FFmpeg pipe with OpenCV fallback.
Runs on RTX 3050 GPU (CUDA) with CPU fallback.
"""
import logging
import cv2
import numpy as np
import subprocess
from typing import Callable, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.gpu_utils import release_gpu_memory
from modules.clip_editor import FFMPEG_EXE

logger = logging.getLogger(__name__)

# COCO class weights. Presence can support framing/context but does not imply a joke.
CONTENT_CLASS_WEIGHTS = {
    0: 2.0,   # person / performer / audience
    56: 0.5,  # chair / seating
    60: 0.4,  # dining table / stage furniture
    65: 0.4,  # bed (generic set prop)
}

_yolo_cache = {}


def _get_yolo():
    key = config.YOLO_MODEL
    if key in _yolo_cache:
        return _yolo_cache[key]
    from ultralytics import YOLO
    model_path = os.path.join(config.BASE_DIR, config.YOLO_MODEL)
    logger.info("Loading YOLO model: %s", model_path)
    m = YOLO(model_path)
    _yolo_cache[key] = m
    return m


def _motion_score(prev_frame, curr_frame) -> float:
    """Frame-difference motion magnitude (0–100)."""
    if prev_frame is None or curr_frame is None:
        return 0.0
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = float(np.mean(gray))
    return min(100.0, score * 4.0)   # scale so 25 mean diff -> 100


def _is_scene_change(prev_frame, curr_frame, threshold=35.0) -> bool:
    """True when a hard scene cut occurs between frames."""
    if prev_frame is None or curr_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, curr_frame)
    return float(np.mean(diff)) > threshold


def _extract_frames_ffmpeg(video_path: str, fps: int = 1, width: int = 640, height: int = 360):
    """Fast frame extraction using FFmpeg pipe scaled directly in C."""
    cmd = [
        FFMPEG_EXE, "-v", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    frame_size = width * height * 3
    second = 0
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
        yield second, frame
        second += 1
    proc.stdout.close()
    proc.wait()


def analyze_visuals(
    video_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[dict]:
    """
    Sample 1 frame per second using hardware/FFmpeg optimized pipeline.
    Returns list of per-second dicts with visual_score, motion, scene_change.
    """
    model = _get_yolo()
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    scores = []
    prev_sampled = None

    # Get duration for progress estimation
    from modules.video_ingestion import probe_video
    vinfo = probe_video(video_path)
    total_sec = max(1.0, float(vinfo.get("duration", 60.0)))

    # 1. Attempt ultra-fast FFmpeg frame pipe
    try:
        for second, small in _extract_frames_ffmpeg(video_path, fps=config.YOLO_SAMPLE_FPS):
            results = model(small, device=device, verbose=False)[0]
            det_score = 0.0
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf > config.YOLO_CONF_THRESHOLD:
                    det_score += CONTENT_CLASS_WEIGHTS.get(cls_id, 0.15)

            motion   = _motion_score(prev_sampled, small)
            scene_ch = _is_scene_change(prev_sampled, small)

            scores.append({
                "second":        second,
                "visual_score":  round(det_score, 2),
                "motion":        round(motion, 2),
                "scene_change":  scene_ch,
            })

            prev_sampled = small
            if progress_callback and total_sec > 0:
                pct = min(100.0, (second / total_sec) * 100.0)
                progress_callback(pct, f"Analyzing performer and scene activity: {pct:.0f}%")

        if len(scores) > 0:
            if progress_callback:
                progress_callback(100.0, "Visual analysis complete")
            logger.info("Visual analysis (FFmpeg pipe): %d seconds processed", len(scores))
            return scores
    except Exception as e:
        logger.warning("FFmpeg frame pipe failed (%s), falling back to OpenCV VideoCapture.", e)

    # 2. Fallback: OpenCV VideoCapture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / config.YOLO_SAMPLE_FPS)))
    frame_idx = 0
    second = 0
    scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            small = cv2.resize(frame, (640, 360))
            results = model(small, device=device, verbose=False)[0]
            det_score = 0.0
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if conf > config.YOLO_CONF_THRESHOLD:
                    det_score += CONTENT_CLASS_WEIGHTS.get(cls_id, 0.15)

            motion   = _motion_score(prev_sampled, small)
            scene_ch = _is_scene_change(prev_sampled, small)

            scores.append({
                "second":        second,
                "visual_score":  round(det_score, 2),
                "motion":        round(motion, 2),
                "scene_change":  scene_ch,
            })
            prev_sampled = small
            second += 1

            if progress_callback and total_sec > 0:
                pct = min(100.0, (second / total_sec) * 100.0)
                progress_callback(pct, f"Analyzing performer and scene activity: {pct:.0f}%")

        frame_idx += 1

    cap.release()
    if progress_callback:
        progress_callback(100.0, "Visual analysis complete")
    return scores


def clear_yolo_cache():
    _yolo_cache.clear()
    release_gpu_memory()

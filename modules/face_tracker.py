"""Robust face-guided portrait framing for webcam, podcast, and reaction clips.

Uses MediaPipe Face Landmarker as primary detector with OpenCV Haar Cascade fallback.
Applies Adaptive Velocity EMA smoothing across temporal samples to eliminate
jerky camera pans and framing jitter in the generated FFmpeg crop expression.
"""
from __future__ import annotations

import logging
import math
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default base directory resolution
try:
    import config
    BASE_DIR = getattr(config, "BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")


def _get_detector(model_path: str = DEFAULT_MODEL_PATH):
    """Instantiate MediaPipe detector if available, otherwise Haar fallback."""
    # Try MediaPipe first
    if os.path.isfile(model_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                num_faces=2,
                min_face_detection_confidence=0.35,
            )
            detector = vision.FaceLandmarker.create_from_options(options)
            return ("mediapipe", detector, mp)
        except Exception as e:
            logger.warning("MediaPipe initialization failed in face_tracker: %s. Trying Haar.", e)

    # Fallback to Haar
    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        haar_data = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if cascade_cls and haar_data:
            cascade_path = os.path.join(haar_data, "haarcascade_frontalface_default.xml")
            if os.path.isfile(cascade_path):
                cascade = cascade_cls(cascade_path)
                if not cascade.empty():
                    return ("haar", cascade, None)
    except Exception as e:
        logger.warning("Haar fallback initialization failed: %s", e)

    return ("none", None, None)


def face_crop_x_expression(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_sec: float = 0.5,
    model_path: str = DEFAULT_MODEL_PATH,
) -> str | None:
    """Return an FFmpeg crop-x expression that follows the primary face with smooth EMA tracking.
    
    Returns None if no faces are detected (downstream callers fall back to centered crop).
    """
    engine_type, detector, mp = _get_detector(model_path)
    if engine_type == "none" or detector is None:
        logger.debug("No face detector available; using centered crop fallback")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    duration = max(1.0, float(end_sec - start_sec))
    raw_points: list[tuple[float, float]] = []  # (rel_time, center_x)
    t = float(start_sec)
    prev_center_x = None

    try:
        while t <= float(end_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += sample_sec
                continue

            h, w = frame.shape[:2]
            detected_x = None

            if engine_type == "mediapipe":
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = detector.detect(mp_img)
                    if res.face_landmarks:
                        # If multiple faces, pick the one closest to previous center or largest
                        candidates = []
                        for lm in res.face_landmarks:
                            xs = [pt.x * w for pt in lm]
                            cx = (min(xs) + max(xs)) / 2.0
                            width_f = max(xs) - min(xs)
                            candidates.append((cx, width_f))
                        if prev_center_x is not None:
                            detected_x = min(candidates, key=lambda c: abs(c[0] - prev_center_x))[0]
                        else:
                            detected_x = max(candidates, key=lambda c: c[1])[0]
                except Exception as e:
                    logger.debug("MediaPipe frame tracking error: %s", e)

            elif engine_type == "haar":
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(
                        gray, scaleFactor=1.12, minNeighbors=5, minSize=(36, 36)
                    )
                    if len(faces) > 0:
                        candidates = []
                        for x, y, fw, fh in faces:
                            cx = x + fw / 2.0
                            candidates.append((cx, fw * fh))
                        if prev_center_x is not None:
                            detected_x = min(candidates, key=lambda c: abs(c[0] - prev_center_x))[0]
                        else:
                            detected_x = max(candidates, key=lambda c: c[1])[0]
                except Exception as e:
                    logger.debug("Haar frame tracking error: %s", e)

            if detected_x is not None:
                raw_points.append((round(t - start_sec, 2), float(detected_x)))
                prev_center_x = detected_x
            elif prev_center_x is not None and len(raw_points) > 0:
                # Brief lost-detection grace period: hold previous center for 1 sample
                if (t - start_sec - raw_points[-1][0]) <= sample_sec * 1.5:
                    raw_points.append((round(t - start_sec, 2), float(prev_center_x)))

            t += sample_sec
    finally:
        cap.release()

    if len(raw_points) < 2:
        return None

    # Apply Adaptive Velocity EMA smoothing across the temporal samples
    smoothed_points: list[tuple[float, float]] = []
    sx = raw_points[0][1]
    smoothed_points.append((raw_points[0][0], round(sx, 1)))

    for i in range(1, len(raw_points)):
        t_rel, rx = raw_points[i]
        diff = abs(rx - sx)
        # Deadzone: 8 pixels at 1080p
        if diff <= 8.0:
            alpha = 0.10  # Heavy smoothing when still
        else:
            factor = min(1.0, (diff - 8.0) / 60.0)
            alpha = 0.20 + factor * 0.45  # Scale up to 0.65 for fast head moves
        sx = alpha * rx + (1.0 - alpha) * sx
        smoothed_points.append((t_rel, round(sx, 1)))

    # Construct bounded FFmpeg piecewise-linear expression
    fallback = "(iw-ow)/2"
    expression = f"max(0,min(iw-ow,{smoothed_points[-1][1]}-ow/2))"

    for idx in range(len(smoothed_points) - 2, -1, -1):
        t0, x0 = smoothed_points[idx]
        t1, x1 = smoothed_points[idx + 1]
        dt = max(0.01, t1 - t0)
        left = f"max(0,min(iw-ow,{x0:.1f}+({x1 - x0:.1f})*(t-{t0:.2f})/{dt:.2f}-ow/2))"
        expression = f"if(lt(t,{t1:.2f}),{left},{expression})"

    if smoothed_points[0][0] > 0:
        expression = f"if(lt(t,{smoothed_points[0][0]:.2f}),{fallback},{expression})"

    return expression.replace(",", r"\,")

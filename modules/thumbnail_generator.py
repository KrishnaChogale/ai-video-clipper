"""
modules/thumbnail_generator.py
Extract the best non-blurry, non-dark frame from a clip for use as a thumbnail.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import cv2
import numpy as np


def _frame_quality(frame: np.ndarray) -> tuple[float, float]:
    """Return (laplacian_variance, mean_brightness) for a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    return lap_var, brightness


def extract_thumbnail(
    video_path: str,
    output_path: str,
    peak_second: float | None = None,
    min_sharpness: float = 100.0,
    min_brightness: float = 40.0,
    sample_every_n_frames: int = 15,
) -> str | None:
    """
    Extract the best-quality frame from a video file and save as JPEG.

    Args:
        video_path:    Path to the source clip (MP4).
        output_path:   Where to save thumbnail.jpg.
        peak_second:   If given, prefer frames near this timestamp (seconds).
        min_sharpness: Laplacian variance threshold; below this = blurry.
        min_brightness: Mean gray value threshold; below this = too dark.
        sample_every_n_frames: Sample 1 frame every N frames for speed.

    Returns:
        output_path if saved successfully, None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Skip first & last 5% of clip (often fade in/out)
    skip_start = int(total_frames * 0.05)
    skip_end   = int(total_frames * 0.95)

    best_frame = None
    best_score = -1.0

    # Bonus for being near peak_second
    peak_frame = int(peak_second * fps) if peak_second is not None else -1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx < skip_start or frame_idx > skip_end:
            continue
        if frame_idx % sample_every_n_frames != 0:
            continue

        lap_var, brightness = _frame_quality(frame)

        if lap_var < min_sharpness or brightness < min_brightness:
            continue

        # Score = sharpness + brightness bonus + proximity bonus
        proximity_bonus = 0.0
        if peak_frame > 0:
            proximity_bonus = max(0.0, 200.0 - abs(frame_idx - peak_frame) * 0.5)

        score = lap_var + brightness * 0.5 + proximity_bonus

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    cap.release()

    if best_frame is None:
        # Fallback: grab middle frame without quality checks
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, best_frame = cap2.read()
        cap2.release()
        if not ret or best_frame is None:
            return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path


def extract_thumbnail_at_second(video_path: str, second: float, output_path: str) -> str | None:
    """Extract the frame at a specific second and save it."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path

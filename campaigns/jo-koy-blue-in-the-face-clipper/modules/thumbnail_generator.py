"""
modules/thumbnail_generator.py
Extract the best non-blurry, non-dark frame from a clip for use as a thumbnail.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import cv2
import numpy as np
import textwrap


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
    sample_every_n_frames: int = 8,
    title: str = "",
    prefer_faces: bool = False,
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
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ) if prefer_faces else None

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

        # Balance detail, usable exposure, peak timing, and (when relevant) a face.
        exposure_score = max(0.0, 35.0 - abs(brightness - 135.0) * 0.35)
        proximity_bonus = 0.0
        if peak_frame > 0:
            proximity_bonus = max(0.0, 24.0 - abs(frame_idx - peak_frame) / max(1.0, fps * 1.5))

        face_bonus = 0.0
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12,
                                                  minNeighbors=5, minSize=(36, 36))
            if len(faces):
                face_bonus = min(30.0, max(w * h for _, _, w, h in faces) /
                                 max(1.0, frame.shape[0] * frame.shape[1]) * 120)

        score = min(100.0, np.log1p(lap_var) * 8.0) + exposure_score + proximity_bonus + face_bonus

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

    if title:
        best_frame = _draw_title(best_frame, title)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def _draw_title(frame: np.ndarray, title: str) -> np.ndarray:
    """Add a readable, high-contrast cover title without obscuring most of the frame."""
    image = frame.copy()
    height, width = image.shape[:2]
    text = title.encode("ascii", "replace").decode("ascii").strip()
    max_chars = max(16, min(34, width // 24))
    lines = textwrap.wrap(text, width=max_chars)[:2]
    if not lines:
        return image
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.55, min(1.2, width / 1000.0))
    thickness = max(1, int(scale * 2.5))
    line_h = int(38 * scale)
    pad = int(22 * scale)
    band_h = min(height, line_h * len(lines) + pad * 2)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - band_h), (width, height), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.68, image, 0.32, 0)
    y = height - band_h + pad + line_h
    for line in lines:
        cv2.putText(image, line, (pad, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, line, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    return image


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

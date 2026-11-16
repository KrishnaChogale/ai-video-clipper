"""
modules/quality_control.py
Evaluate a clip against quality criteria before export.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import cv2
import numpy as np


def evaluate_clip(
    clip_path: str,
    start_sec: float,
    end_sec: float,
    score: float,
    segments: list | None = None,
    min_score: float = 20.0,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> dict:
    """
    Run quality checks on a produced clip file.

    Returns:
        {
            "passed": bool,
            "issues": [str],
            "auto_fixable": bool,
            "details": {
                "duration": float,
                "first_frame_ok": bool,
                "audio_ok": bool,
                "score_ok": bool,
            }
        }
    """
    issues = []
    auto_fixable = True

    min_dur = min_duration or config.CLIP_MIN_DURATION
    max_dur = max_duration or config.CLIP_MAX_DURATION

    duration = end_sec - start_sec
    dur_ok = min_dur <= duration <= max_dur
    if not dur_ok:
        issues.append(f"Duration {duration:.1f}s outside [{min_dur}–{max_dur}]s bounds")
        auto_fixable = False

    score_ok = score >= min_score
    if not score_ok:
        issues.append(f"Highlight score {score:.1f} below minimum {min_score}")

    first_frame_ok = True
    audio_ok = True

    if os.path.isfile(clip_path):
        cap = cv2.VideoCapture(clip_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                if mean_brightness < 10:
                    first_frame_ok = False
                    issues.append(f"First frame too dark (brightness={mean_brightness:.1f})")
            else:
                first_frame_ok = False
                issues.append("Could not read first frame from clip file")
            cap.release()
        else:
            first_frame_ok = False
            issues.append("Could not open clip file with OpenCV")

        # Basic audio check: file must be > 10 KB (not empty / encoding error)
        file_size = os.path.getsize(clip_path)
        if file_size < 10_000:
            audio_ok = False
            issues.append(f"Clip file suspiciously small ({file_size} bytes)")
    else:
        issues.append(f"Clip file not found: {clip_path}")
        auto_fixable = False

    passed = len(issues) == 0

    return {
        "passed": passed,
        "issues": issues,
        "auto_fixable": auto_fixable,
        "details": {
            "duration": duration,
            "first_frame_ok": first_frame_ok,
            "audio_ok": audio_ok,
            "score_ok": score_ok,
        },
    }

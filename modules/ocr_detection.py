"""
modules/ocr_detection.py
OCR detection for on-screen text (mission titles, wanted stars, HUD indicators).
Gracefully falls back if Tesseract binary is not installed on the system.
"""
import logging
import cv2
import numpy as np
import shutil
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

_tesseract_configured = None


def _init_tesseract():
    global _tesseract_configured
    if _tesseract_configured is not None:
        return _tesseract_configured

    try:
        import pytesseract
        # 1. Check config.TESSERACT_CMD
        if config.TESSERACT_CMD and os.path.isfile(config.TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            _tesseract_configured = True
            return True

        # 2. Check system PATH
        which_tess = shutil.which("tesseract")
        if which_tess:
            pytesseract.pytesseract.tesseract_cmd = which_tess
            _tesseract_configured = True
            return True

        # 3. Known standard Windows 64-bit path
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win
            _tesseract_configured = True
            return True

    except Exception as e:
        logger.debug("Tesseract initialization failed: %s", e)

    _tesseract_configured = False
    return False


def detect_screen_text(frame: np.ndarray) -> str:
    """
    Extract readable on-screen text from a video frame.
    Returns empty string if Tesseract is not installed or no text found.
    """
    if not _init_tesseract():
        return ""

    try:
        import pytesseract
        # Preprocessing: crop to top and bottom banners (where HUD/wanted stars/mission text appear)
        h, w = frame.shape[:2]
        # Crop top 25% (wanted stars, location) and bottom 20% (subtitles, mission prompt)
        top_banner = frame[0:int(h * 0.25), :]
        gray = cv2.cvtColor(top_banner, cv2.COLOR_BGR2GRAY)
        # Thresholding for HUD text
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        return text.strip()
    except Exception as e:
        logger.debug("OCR frame extraction error: %s", e)
        return ""

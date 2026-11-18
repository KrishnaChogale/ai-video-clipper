"""
Jo Koy: Blue in the Face Campaign Clipper — Central Configuration
All tuneable settings live here. Never hardcode paths or thresholds in module files.
"""
import os
import sys

# Ensure PyTorch's bundled CUDA libraries (cublas64_12.dll, etc.) are in DLL search path on Windows
try:
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + ";" + os.environ.get("PATH", "")
except Exception:
    pass

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR     = os.path.join(BASE_DIR, "input")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
TEMP_DIR      = os.path.join(BASE_DIR, "temp")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
DATABASE_DIR  = os.path.join(BASE_DIR, "database")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")   # legacy compat
CLIPS_DIR     = os.path.join(BASE_DIR, "clips")        # legacy compat

for _d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, MODELS_DIR, DATABASE_DIR, LOGS_DIR, DOWNLOADS_DIR, CLIPS_DIR]:
    os.makedirs(_d, exist_ok=True)

DB_PATH = os.path.join(DATABASE_DIR, "clips.db")

# ─────────────────────────── GPU / COMPUTE ───────────────────────────
# Preferred device: "auto" | "cuda" | "cpu"
PREFERRED_DEVICE = "auto"
# Max VRAM to use for model loading (GB). Keep headroom for other processes.
MAX_VRAM_GB = 4.0

# ─────────────────────────── WHISPER ───────────────────────────
# tiny | base | small | medium (larger = more accurate, slower)
WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE_CUDA = "float16"
WHISPER_COMPUTE_TYPE_CPU  = "int8"
WHISPER_BEAM_SIZE = 5

# ─────────────────────────── YOLO ───────────────────────────
YOLO_MODEL = "yolo11n.pt"           # lightweight nano model
YOLO_CONF_THRESHOLD = 0.40
# Frames sampled per second for YOLO (lower = faster, less accurate)
YOLO_SAMPLE_FPS = 1

# ─────────────────────────── CLIP SETTINGS ───────────────────────────
CLIP_MIN_DURATION = 15    # seconds
CLIP_MAX_DURATION = 180   # seconds
CLIP_VIRAL_MIN_DURATION = 30  # seconds (sweet spot for Reels/Shorts)
CLIP_VIRAL_MAX_DURATION = 45  # seconds
CLIP_DEFAULT_DURATION = 35    # default 35s (within 30-45s viral range)
CLIP_DEFAULT_COUNT = 5        # number of clips to generate
CLIP_TOP_K_MAX = 20           # hard cap on clips per run
MIN_VIRALITY_SCORE = 70        # Candidate score floor; heuristic, not a probability
POV_FORMAT_FRACTION = 0.20     # Apply POV opening style to the top fifth of selected clips

# ─────────────────────────── DEAD AIR & JUMP-CUTS ───────────────────────────
REMOVE_DEAD_AIR_DEFAULT = True
DEAD_AIR_MIN_PAUSE_SEC = 0.45    # Pauses longer than this are trimmed (0.45s)
DEAD_AIR_NOISE_THRESH_DB = -32.0 # Audio silence threshold in dB
DEAD_AIR_PAD_SEC = 0.06          # Natural breath padding on speech cuts

# ─────────────────────────── HIGHLIGHT SCORING WEIGHTS ───────────────────────────
# Must sum to 100
SCORE_WEIGHTS = {
    "visual_action":    25,
    "audio_excitement": 20,
    "unexpected_event": 20,
    "story_context":    15,
    "event_clarity":    10,
    "payoff_potential": 10,
}

# Evidence factors are bounded independently so audio peaks cannot dominate.
HIGHLIGHT_FACTOR_WEIGHTS = {
    "event_significance": 20,
    "event_clarity": 8,
    "visual_action": 8,
    "audio_excitement": 16,
    "unexpectedness": 10,
    "build_up_payoff": 15,
    "story_context": 15,
    "replay_potential": 4,
    "video_quality": 4,
}

# ─────────────────────────── EXPORT / FFMPEG ───────────────────────────
# Output resolution for vertical 9:16 shorts
EXPORT_WIDTH  = 1080
EXPORT_HEIGHT = 1920
EXPORT_FPS    = 30
EXPORT_CRF    = 22          # H.264 quality: 18 (best) – 28 (worst)
EXPORT_PRESET = "fast"      # ultrafast | fast | medium | slow
EXPORT_AUDIO_BITRATE = "128k"

# Try NVENC hardware encoding first; falls back automatically
USE_NVENC = True

# ─────────────────────────── CAPTION SETTINGS ───────────────────────────
CAPTION_FONT_SIZE_VERTICAL  = 68
CAPTION_FONT_SIZE_HORIZONTAL = 48
CAPTION_OUTLINE = 5.0
CAPTION_SHADOW  = 2.0
CAPTION_COLOR_HIGHLIGHT = "&H0000FFFF&"   # Vibrant Yellow active word highlight
CAPTION_COLOR_BASE      = "&H00FFFFFF&"   # Crisp White text
CAPTION_COLOR_OUTLINE   = "&H00000000&"   # Jet Black outline
CAPTION_COLOR           = "&H0000FFFF"    # Legacy yellow ASS colour
CAPTION_MAX_WORDS_PER_CUE = 3             # 2-3 words per cue for viral kinetic pacing
CAPTION_DYNAMIC_HIGHLIGHT_DEFAULT = True  # Word-by-word active pop
CAPTION_SAFE_MARGIN_V_REELS = 440         # Above 350px Reels/Shorts bottom overlay
CAPTION_SAFE_MARGIN_H_REELS = 80          # Safe from right side interaction icons
CENSOR_PROFANITY_DEFAULT = True

# ─────────────────────────── METADATA / LLM ───────────────────────────
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 30          # seconds
HASHTAG_COUNT_DEFAULT = 5
# Platforms for description generation
PLATFORMS = ["YouTube Shorts", "Instagram Reels", "TikTok"]

# ─────────────────────────── OCR ───────────────────────────
# Set to Tesseract executable path if not on system PATH (Windows)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────── AUDIO ANALYSIS ───────────────────────────
AUDIO_SAMPLE_RATE = 16000
HOP_LENGTH = 512

# ─────────────────────── COMEDY CAMPAIGN SIGNALS ─────────────────────────
# These are scoring hints, not claims that a joke or audience reaction occurred.
COMEDY_EVENT_BONUS = {
    "explicit_laughter_cue": 28,
    "possible_audience_reaction": 12,
    "speech_payoff_candidate": 10,
    "sustained_performance": 5,
}

CAMPAIGN_NAME = "Jo Koy: Blue in the Face"
CAMPAIGN_HASHTAGS = ["#JoKoy", "#BlueInTheFace", "#StandUpComedy", "#ComedyClips"]

# ─────────────────────────── UPLOAD LIMIT ───────────────────────────
# In MB (Streamlit config also set in .streamlit/config.toml)
MAX_UPLOAD_MB = 51200   # 50 GB

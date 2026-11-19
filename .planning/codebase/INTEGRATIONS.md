# External Integrations
**Analysis Date:** 2026-11-18

## APIs & External Services

**Video & Livestream Platforms:**
- YouTube (VODs & Livestreams) - Download full videos and capture live segments
  - Client: `yt-dlp` library (`modules/video_ingestion.py`)
  - Auth: Netscape-formatted cookie files (`uploaded_cookies.txt`, `ai-video-clippercookies.txt.txt`, or local browser cookie extraction)
- Twitch & Kick - Livestream stream ingestion via HLS stream extraction and FFmpeg pipe
  - Client: `yt-dlp` + FFmpeg (`modules/video_ingestion.py`)
  - Auth: Public streams or cookie-based

**Local LLM Inference:**
- Ollama API (`http://localhost:11434`) - Automated title, hook, description, and hashtag generation
  - Client: `ollama` Python SDK (`modules/title_generator.py`, `modules/description_generator.py`)
  - Model: `llama3.2:latest` (with rule-based algorithmic fallbacks when offline or on timeout)
  - Auth: None (local daemon)

## Data Storage

**Databases:**
- SQLite 3
  - Connection: `database/clips.db` (configured via `config.DB_PATH`)
  - Client: Python standard library `sqlite3` (`modules/metadata_manager.py`)
  - Tables: `clips` (metadata, timestamps, scores, paths), `clip_performance` (views, likes, shares, comments)

**File Storage:**
- Local filesystem only:
  - Input videos: `input/`
  - Rendered clip bundles: `output/{clip_id}/` (video `clip.mp4`, thumbnail `thumbnail.jpg`, subtitles `.ass` / `.srt`, metadata `.json`)
  - Intermediate scratch files: `temp/`
  - Model checkpoints: `models/` (and root `yolo11n.pt`)
  - Logs: `logs/`

**Caching:**
- In-memory Python process caches:
  - `modules/speech_recognition.py`: `_model_cache` dictionary caching loaded Whisper models by `(model_size, device)`
  - `modules/object_detection.py`: `_yolo_cache` dictionary caching loaded YOLO model instances
  - `modules/clip_editor.py`: `_NVENC_AVAILABLE` memoized hardware encoding availability check

## Authentication & Identity

**Auth Provider:**
- Custom / Local: No user accounts or web authentication required (local desktop Streamlit tool).
- YouTube Platform Auth: Handled via cookies file upload in Streamlit UI to bypass YouTube 429/bot rate limits and age gates.

## Monitoring & Observability

**Error Tracking:**
- Local Python standard `logging` with module-level loggers (`logging.getLogger(__name__)`).
- UI error reporting via Streamlit alerts (`st.error`, `st.warning`, `st.info`).

**Logs:**
- Standard stderr/stdout logging and `logs/` directory.

## CI/CD & Deployment

**Hosting:**
- Local execution on Windows workstation (`.\venv\Scripts\python.exe -m streamlit run app.py`).
- No cloud hosting or remote server deployment.

**CI Pipeline:**
- None detected (local Git repository synchronized with GitHub remote `https://github.com/KrishnaChogale/ai-video-clipper.git`).

## Environment Configuration

**Required Configurations:**
- `config.py` tuneable constants (VRAM caps, model names, factor weights, output dimensions).
- Tesseract OCR binary path: `C:\Program Files\Tesseract-OCR\tesseract.exe` (optional for HUD detection).
- Windows TrueType Font: `C:/Windows/Fonts/arialbd.ttf` (used by FFmpeg `drawtext`).

**Secrets Location:**
- No cloud API keys or secret tokens required.
- YouTube cookies file stored at `uploaded_cookies.txt` or `ai-video-clippercookies.txt.txt` (local file).

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None (completely self-contained local processing pipeline).

---

*Integration audit: 2026-11-18*

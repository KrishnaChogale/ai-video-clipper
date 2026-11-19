# Technology Stack
**Analysis Date:** 2026-11-18

## Languages

**Primary:**
- Python 3.12.10 - Core application logic, modules, Streamlit UI, tests (`app.py`, `modules/*.py`, `tests/*.py`)

**Secondary:**
- Advanced SubStation Alpha (`.ass`) & SRT - Dynamic karaoke subtitle scripts with custom styles and tags (`modules/subtitle_generator.py`)
- SQL (SQLite dialect) - Clip metadata, score factors, and analytics persistence (`modules/metadata_manager.py`, `database/clips.db`)
- FFmpeg Filtergraph Syntax - Custom video/audio processing filter complex expressions (`modules/clip_editor.py`, `modules/face_tracker.py`)

## Runtime

**Environment:**
- Python 3.12.10 (Virtual environment at `c:\ai-video-clipper\venv`)
- Windows 11 Home 64-bit
- CUDA 12.1 / PyTorch `2.5.1+cu121` on NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)
- FFmpeg 9.0.2-full_build (Gyan Windows build with `h264_nvenc` and `hevc_nvenc`)

**Package Manager:**
- `pip`
- Lockfile: Missing (`requirements.txt` present without strict hash lockfile)

## Frameworks

**Core:**
- Streamlit (>= 1.40.0) - Interactive web UI, pipeline step runner, real-time progress indicators (`app.py`)

**Testing:**
- Python standard `unittest` (`tests/test_viral_features.py`, `tests/test_modules.py`, `tests/test_caption_templates.py`)

**Build/Dev:**
- Windows PowerShell & batch scripting (`venv/Scripts/python.exe`)

## Key Dependencies

**Critical:**
- `faster-whisper` (>= 1.0.0) - CTranslate2-based Whisper speech-to-text with FP16 CUDA acceleration and word-level timestamps (`modules/speech_recognition.py`)
- `ultralytics` (>= 8.0.0) - YOLO11 nano model (`yolo11n.pt`) for real-time video object detection (`modules/object_detection.py`)
- `opencv-python` (>= 4.8.0, < 5.0) - Image processing, frame difference motion detection, and Haar cascade face/smile detection (`modules/facecam.py`, `modules/face_tracker.py`, `modules/object_detection.py`)
- `librosa` (>= 0.10.0) & `soundfile` (>= 0.12.0) - Audio RMS energy, onset transient detection, and acoustic signal feature extraction (`modules/audio_analysis.py`)
- `yt-dlp` (>= 2024.10.0) - Video downloading from YouTube and real-time live stream HLS/DASH segment capture (`modules/video_ingestion.py`)
- `ollama` (>= 0.3.0) - Local LLM client communicating with Ollama running `llama3.2` for viral hooks, titles, and descriptions (`modules/title_generator.py`, `modules/description_generator.py`)

**Infrastructure:**
- `psutil` (>= 5.9.0) - System memory and process monitoring
- `pytesseract` (>= 0.3.10) - OCR text extraction from gameplay HUDs (`modules/ocr_detection.py`)
- `pandas` (>= 2.0.0) & `numpy` (>= 1.24.0) - Data wrangling and vector math for score distributions
- `scipy` (>= 1.10.0) - Signal processing utilities

## Configuration

**Environment:**
- Configured centrally in `config.py`
- Dynamic DLL directory injection: `torch/lib` is added to Windows DLL search path via `os.add_dll_directory` and `PATH` to resolve `cublas64_12.dll` and cuDNN dependencies for Faster-Whisper and PyTorch
- Streamlit configuration in `.streamlit/config.toml` (`maxUploadSize = 51200` MB for 50GB video ingestion)

**Build:**
- `config.py` defines export dimensions (1080x1920 vertical), FPS (30), CRF (22), and codec preferences (NVENC first, fallback to CPU `libx264`)

## Platform Requirements

**Development:**
- Windows 11 with NVIDIA GPU (minimum 4GB VRAM recommended, tested on RTX 3050 6GB)
- CUDA 12.1 runtime libraries bundled with PyTorch
- External FFmpeg binary installed or packaged via WinGet (`Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe`)
- Local Ollama daemon running on `http://localhost:11434` with `llama3.2` model pulled

**Production:**
- Local workstation / on-premise GPU rig (designed for 100% free offline local processing without cloud SaaS API fees)

---

*Stack analysis: 2026-11-18*

# GTA 6 AI Auto Clipper — Project Implementation & Progress Log

**Location:** `C:\ai-video-clipper`  
**Environment:** Windows 11 · Python 3.12.10 (`.\venv`) · NVIDIA RTX 3050 6GB Laptop GPU  
**Hardware Acceleration:** PyTorch `2.5.1+cu121` (CUDA 12.1 Active) · NVIDIA NVENC Hardware Video Encoding  
**Interface:** Streamlit Web Application (`app.py`)  
**Version:** 2.0.0 (Modular Architecture)  
**Date:** November 16, 2026  

---

## 1. Executive Summary

The **GTA 6 AI Auto Clipper** is an intelligent, locally-running gaming content creation studio similar to Opus Clip, tailored specifically for GTA 6 and action gameplay footage.

It performs end-to-end multimodal gameplay analysis combining:
1. **Audio Energy & Spikes:** Detects gunshots, explosions, sirens, and sudden excitement using Librosa 16kHz mono RMS and onset strength.
2. **Speech Recognition:** Transcribes dialogue with word-level timestamps using Faster-Whisper on CUDA with profanity masking.
3. **Visual Object & Motion Detection:** Tracks vehicles, characters, motion magnitude, and hard scene cuts using Ultralytics YOLO nano and OpenCV.
4. **GTA Event Classification:** Tags police chases, high-speed driving, crashes, shootouts, explosions, stunts, funny NPC reactions, and cinematic views.
5. **Multi-Modal Highlight Scoring:** Evaluates candidate moments on a 0–100 scale across 6 configurable factors.
6. **AI Metadata & Retention Hooks:** Generates 3 alternative viral titles, 3 opening hooks, platform-specific descriptions (YouTube Shorts, Instagram Reels, TikTok), and relevant hashtags using local Ollama (`llama3.2`).
7. **Viral Subtitles & Safe Zones:** Generates styled `.ass` and `.srt` subtitles with animated 3–4 word cues placed in lower-third safe zones (clearing mobile UI navigation).
8. **Vertical 9:16 Editing & NVENC Export:** Produces vertical clips (1080x1920) via NVIDIA NVENC hardware acceleration with center-crop or blurred-background framing.
9. **Automated Quality Control:** Evaluates duration, first-frame clarity, audio integrity, and score thresholds before exporting.
10. **Organised Clip Bundles & Performance Tracking:** Exports each clip into `output/{clip_id}/` (video, thumbnail, subtitles, metadata, titles, descriptions) and tracks real-world views/likes in SQLite.

---

## 2. Modular Architecture Overview

```
C:\ai-video-clipper\
│
├── app.py                     # Streamlit web interface and orchestrator
├── config.py                  # Centralized configuration (paths, models, weights)
├── PROJECT_PROGRESS.md        # Comprehensive technical documentation
│
├── modules/
│   ├── __init__.py
│   ├── gpu_utils.py           # GPU VRAM detection, CUDA status, memory release
│   ├── video_ingestion.py     # YouTube download (cookies/auth) + local file handler
│   ├── audio_analysis.py      # Librosa energy buckets, onset transients, explosions
│   ├── speech_recognition.py  # Faster-Whisper on CUDA with word-level timestamps
│   ├── object_detection.py    # YOLO nano detection + OpenCV motion tracking
│   ├── event_classification.py# GTA event tagging (police chase, crash, stunts)
│   ├── highlight_scoring.py   # 6-factor highlight scoring (0–100) & windowing
│   ├── title_generator.py     # 3 title options + 3 hook options (Ollama/templates)
│   ├── description_generator.py# Platform descriptions (YT/IG/TT) + hashtags
│   ├── subtitle_generator.py  # ASS & SRT captions, profanity masking, safe zones
│   ├── clip_editor.py         # FFmpeg NVENC/libx264 cutter, 9:16 crop/blur-bg
│   ├── thumbnail_generator.py # Sharpness/brightness cover frame extractor
│   ├── quality_control.py     # Pre-export technical and retention checklist
│   ├── ocr_detection.py       # HUD and on-screen text detection
│   └── metadata_manager.py    # SQLite database + JSON metadata exporter
│
├── tests/
│   └── test_modules.py        # 9 unit & integration tests covering all modules
├── database/                  # SQLite database (database/clips.db)
├── input/                     # Downloaded and staged video files
├── output/                    # Exported clip bundles (output/clip_XX_.../)
└── temp/                      # Intermediate scratch files
```

---

## 3. Hardware & Environment Audit

- **Operating System:** Windows 11 Home 64-bit
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver 596.08, CUDA 12.1 supported)
- **PyTorch:** `2.5.1+cu121` (`torch.cuda.is_available() == True`, ~5.4 GB free VRAM available at runtime)
- **FFmpeg:** Gyan FFmpeg full build with `h264_nvenc` hardware acceleration verified
- **Ollama:** Local Ollama running `llama3.2:latest`
- **Execution Rule:** Always run via `.\venv\Scripts\python.exe -m ...` due to Windows Application Control policies.

---

## 4. Key Upgrades Implemented

### A. CUDA PyTorch 2.5.1 Installation & VRAM Optimization
- Successfully upgraded from PyTorch CPU-only (`2.15.0+cpu`) to `torch==2.5.1+cu121` and `torchvision==0.20.1+cu121`.
- Both YOLO nano and Faster-Whisper run natively on the RTX 3050 GPU, yielding ~5x faster processing.
- Added explicit `release_gpu_memory()` calls (`torch.cuda.empty_cache()` and `ipc_collect()`) between pipeline stages to keep memory usage low.

### B. High-Speed NVENC Hardware Video Encoding
- Verified `h264_nvenc` availability in FFmpeg.
- Slicing and exporting vertical 1080x1920 clips with burned ASS subtitles renders in seconds per clip.
- Automatic fallback to `libx264` ensures stability if hardware encoding encounters unsupported parameters.

### C. GTA Gameplay Event Classification
- Heuristic multi-modal event classifier detects:
  - **Action:** Gunshots, explosions, police chases, car crashes, stunts, high-speed driving.
  - **Funny:** NPC unusual reactions and dialogue surprises.
  - **Cinematic:** Scenic vistas, environmental sunsets, character pauses with high visual interest.
  - **Discovery:** Hard scene transitions, game mechanics, and environmental details.

### D. Automated Title, Hook, and Description Generation
- Generates 3 alternative titles per clip: curiosity-based, action-based, and reaction-based.
- Generates 3 on-screen hook text options (0–2s viewer hook).
- Generates 3 tailored descriptions for YouTube Shorts, Instagram Reels, and TikTok.
- Generates 3–5 contextual hashtags per clip.

### E. Social Media Performance Tracking
- Added a dedicated **Performance Tracking** tab in the Streamlit UI.
- Stores Views, Likes, Shares, Comments, Avg Watch Time, Completion Rate, and Subscribers Gained in `database/clips.db` (`clip_performance` table).
- Allows content creators to measure which video formats and hooks deliver the highest retention.

---

## 5. Verification & Test Suite

All 9 test suites in `tests/test_modules.py` executed cleanly:
- `test_gpu_utils`: Verified CUDA detection, VRAM metrics, and device selection.
- `test_censor_profanity`: Verified regex masking of swear words (e.g. `F*CK`, `SH*T`, `MOTHERF*CKER`).
- `test_ass_and_srt_formatting`: Verified ASS canvas sizing, lower-third placement, and SRT timestamps.
- `test_audio_energy_buckets`: Verified RMS loudness and onset spike calculations on audio arrays.
- `test_highlight_scoring_and_selection`: Verified 6-factor scoring and non-overlapping clip extraction.
- `test_title_and_hook_generation`: Verified 3 titles and 3 hooks generated per clip.
- `test_description_and_hashtag_generation`: Verified multi-platform descriptions and hashtag pools.
- `test_quality_control`: Verified duration, brightness, and audio check logic.
- `test_database_and_metadata`: Verified SQLite insertion, JSON export, and performance tracking.

---

## 6. How to Run

```powershell
cd C:\ai-video-clipper
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

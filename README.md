# 🎬 GTA 6 AI Auto Clipper

> **Intelligent, local short-form content generator for GTA 6 gameplay — inspired by Opus Clip, built for local GPUs.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20Supported-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source, self-hosted AI pipeline that ingests long GTA 6 gameplay or trailers, analyzes speech, audio transients, and visual action, and automatically cuts, frames, and captions viral 9:16 vertical shorts (YouTube Shorts, TikTok, Instagram Reels) — running 100% locally on consumer hardware (tested on NVIDIA RTX 3050 6GB).

---

## ⚡ Key Features

- **🎯 Multi-Modal Highlight Scoring (0–100):** Combines audio onset energy spikes, YOLO object detection (police cars, guns, explosions), speech transcription, and visual motion to select peak viral moments.
- **🎙️ Fast Local Subtitles:** Uses `faster-whisper` (FP16 CUDA) to generate word-level timed subtitles with custom animated styles (.ass/.srt) and profanity filtering.
- **📱 Smart 9:16 Vertical Reframing:** Auto-crops horizontal 16:9 gameplay to vertical 9:16 with blurred sidebars or dynamic centering.
- **💡 AI Viral Titles & Hooks:** Generates catchy titles and opening hooks using local Ollama (`llama3.2`) with instant heuristic fallbacks.
- **🖼️ Sharp Cover Thumbnail Extraction:** Automatically extracts high-contrast, sharpest keyframes using Laplacian variance.
- **🖥️ Beautiful Streamlit Dashboard:** User-friendly UI for single/batch processing, real-time log monitoring, clip history, and analytics.
- **🚀 6GB VRAM & Laptop GPU Optimized:** Modular sequential loading, automatic VRAM cleanup, fast FFmpeg C-pipe frame decoding, and graceful CPU/libx264 fallbacks.

---

## 🛠️ System Requirements

- **OS:** Windows 10/11 or Linux
- **GPU:** NVIDIA GPU with 4GB–6GB+ VRAM (RTX 3050/3060/4050 or higher recommended)
- **RAM:** 8 GB minimum (16 GB recommended)
- **External Tools:**
  - [FFmpeg](https://ffmpeg.org/) (must be on system `PATH`)
  - [Ollama](https://ollama.com/) (optional, for local AI title/hook generation)

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ai-video-clipper.git
cd ai-video-clipper
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Launch the Application
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Architecture

```
ai-video-clipper/
├── app.py                      # Main Streamlit web application
├── config.py                   # Centralized configuration & CUDA DLL handler
├── requirements.txt            # Python dependencies
├── modules/                    # Modular processing pipeline
│   ├── gpu_utils.py            # VRAM detection & memory release
│   ├── video_ingestion.py      # Video download (yt-dlp) & ffprobe inspection
│   ├── audio_analysis.py       # Librosa RMS loudness & onset spike detection
│   ├── speech_recognition.py   # faster-whisper word-level transcription
│   ├── object_detection.py     # YOLO object tracking & scene change detection
│   ├── event_classification.py # GTA event heuristics (police, crashes, stunts)
│   ├── highlight_scoring.py    # 6-factor multi-modal score aggregator
│   ├── clip_editor.py          # FFmpeg 9:16 vertical cutter with NVENC/libx264
│   ├── subtitle_generator.py   # Word-level ASS subtitles & profanity censoring
│   ├── title_generator.py      # Ollama/heuristic viral hook & title generator
│   ├── description_generator.py# Platform description & hashtag generator
│   ├── thumbnail_generator.py  # Sharpest frame extractor
│   ├── ocr_detection.py        # HUD text & on-screen killfeed OCR
│   ├── quality_control.py      # Pre-export audio/video duration sanity checks
│   └── metadata_manager.py     # SQLite persistence & social tracking
└── tests/                      # Verification and stress test suites
```

---

## 🔒 Security & Privacy Notice

- Private authentication tokens or YouTube cookies (`cookies.txt`, `uploaded_cookies.txt`) are ignored by `.gitignore` and must **never** be committed to public repositories.
- All AI processing (transcription, object detection, LLM generation) runs **100% locally** on your machine with zero data sent to external cloud APIs.

---

## 📄 License
MIT License. Free for personal and commercial use.

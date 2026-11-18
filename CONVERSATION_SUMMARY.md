# 🎮 GTA 6 AI Auto Clipper — Conversation & Project Record

> **Generated for Krishna Chogale** | Project: Local AI Short-Form Content Generator  
> **Environment:** Windows 11 | NVIDIA RTX 3050 Laptop GPU (6.44 GB VRAM) | Python 3.12 | Local-Only & 100% Free

---

## 📌 1. Project Overview & Architecture
This project is an open-source, locally running AI video clipper (inspired by Opus Clip, CapCut, and Munch) specialized for GTA 6 and high-energy gameplay. It automatically takes long-form gameplay videos and turns them into viral 30–45s vertical shorts for YouTube Shorts, Instagram Reels, and TikTok.

### Core Modules (`C:\ai-video-clipper\modules\`):
1. **`speech_recognition.py`**: Faster-Whisper with word-level timestamps (CUDA FP16 / CPU fallback).
2. **`object_detection.py`**: YOLOv8 nano for real-time car/person/vehicle detection via FFmpeg pipe.
3. **`audio_analysis.py`**: Energy spikes, action bucket scoring, and speech activity detection.
4. **`highlight_scoring.py`**: Multi-modal scoring with viral 30–45s duration targeting and opening 3s hook optimization.
5. **`clip_editor.py`**: FFmpeg video cutting, dead air removal (jump cuts), blurred background 9:16 layout, and NVENC/CPU encoding.
6. **`subtitle_generator.py`**: 
   - Dynamic kinetic word-by-word highlighting (active yellow pop on base white text).
   - Safe zone margins (440px from bottom to clear mobile app UI).
   - Profanity censoring (`F*CK`, `SH*T`).
   - Context-aware subtitle emojis (🔥, 💀, 🚗, 🚨, 🎯, 💥, 💰, 😂).
   - On-screen Hook Title Banner in top safe zone for the first 3.5 seconds.
7. **`caption_templates.py`**: Style presets (Gaming Style, MrBeast Style, Neon, Minimal, GTA Vice City).
8. **`face_tracker.py`**: Face tracking and auto-reframing for vertical crop.
9. **`title_generator.py`**: Ollama local LLM (`llama3.2`) + rule-based fallback for catchy titles and hooks.
10. **`description_generator.py`**: Platform-tailored descriptions (Shorts, Reels, TikTok) and trending hashtags.
11. **`thumbnail_generator.py`**: High-energy peak frame extractor with title overlay.
12. **`quality_control.py`**: Black-frame, blur, silence, and duration technical validation.
13. **`metadata_manager.py`**: SQLite database storing clip records, timestamps, scores, and performance history.

---

## 🎬 2. Real-World Tested Features
- **Duration:** Exactly 30.00s short-form cuts.
- **Visuals:** 1080x1920 (9:16) with blurred gameplay background top & bottom.
- **Hook Optimization:** 
  - Opening 3s ranked with action onset.
  - Top Hook Banner rendered during first 3.5s (`YOU NEED TO SEE THIS 🔥`).
- **Dead Air Removal:** Automatically removes silent pauses to keep high viewer retention.
- **Subtitles:** Frame-accurate kinetic karaoke captions that stay synced across jump-cuts.

---

## 🚀 3. How to Launch the Application
Open PowerShell and run:
```powershell
cd C:\ai-video-clipper
.\venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 4. How to Run Test Suites
```powershell
cd C:\ai-video-clipper
$env:PYTHONPATH="C:\ai-video-clipper"
.\venv\Scripts\python.exe tests/test_viral_features.py
.\venv\Scripts\python.exe tests/test_modules.py
.\venv\Scripts\python.exe tests/test_caption_templates.py
```

---

## 🌿 5. Git Status & Policy
- **Remote:** `https://github.com/KrishnaChogale/ai-video-clipper.git`
- **Active Branch:** `Krishna` (synchronized with `main`)
- **Important Policy Rule:** Never commit or push to Git/GitHub without explicit user confirmation first.

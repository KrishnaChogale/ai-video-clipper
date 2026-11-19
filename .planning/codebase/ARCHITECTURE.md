<!-- refreshed: 2026-11-18 -->
# Architecture
**Analysis Date:** 2026-11-18

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Streamlit User Interface (`app.py`)                   │
├──────────────────┬──────────────────┬───────────────────────┬───────────────┤
│ YouTube / Upload │  Audio Analysis  │   Visual Detection    │ Highlight &   │
│ Ingestion        │  & Transcription │   & Facecam Tracking  │ Title Studio  │
│ `video_ingestion`│ `audio_analysis` │   `object_detection`  │ `highlight_   │
│                  │ `speech_recog`   │   `facecam.py`        │  scoring.py`  │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┴───────┬───────┘
         │                  │                    │                    │
         ▼                  ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Modal Evidence Fusion & Windows                     │
│  - 10-Factor Highlight Scoring & Opening Hook Optimizer (`highlight_scoring.py`)│
│  - Conservative Event Classification (`event_classification.py`)            │
│  - Facecam Quota & Reaction Ranking (`facecam.py`, `app.py`)                 │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Rendering & Editing Engine                              │
│  - Dynamic Word-by-Word ASS Subtitle Generator (`subtitle_generator.py`)    │
│  - Dead-Air Jump Cut Detection (`clip_editor.py`)                           │
│  - Facecam Split (Top Facecam / Bottom Gameplay) (`_cut_facecam_split`)     │
│  - NVENC / libx264 Hardware Video Pipeline (`clip_editor.py`)               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Output Bundles & Persistence Layer                          │
│  - Package: `output/{clip_id}/` (`clip.mp4`, thumbnail, subtitles, metadata) │
│  - Quality Control Validator (`quality_control.py`)                         │
│  - SQLite Analytics Database (`database/clips.db`, `metadata_manager.py`)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Streamlit Orchestrator** | Coordinates 5-step analysis pipeline, maintains session state, renders dashboard UI | `app.py` |
| **Central Config** | Declares paths, VRAM limits, factor weights, ASS styling presets, export resolutions | `config.py` |
| **Video Ingestion** | Handles YouTube downloads via `yt-dlp`, live stream HLS capture, local uploads, and `ffprobe` metadata | `modules/video_ingestion.py` |
| **Audio Analysis** | Computes 16kHz mono RMS loudness, onset strength transients, and gunshot/explosion flags | `modules/audio_analysis.py` |
| **Speech Recognition** | Transcribes dialogue via Faster-Whisper on CUDA FP16 with word-level timestamps | `modules/speech_recognition.py` |
| **Visual Detection** | Samples 1 FPS frames via FFmpeg pipe, runs YOLO11 nano detection, frame-diff motion, and scene cuts | `modules/object_detection.py` |
| **Facecam Tracking** | Identifies persistent corner webcam insets via Haar cascades, tracks smile/motion reaction proxies | `modules/facecam.py` |
| **Face Portrait Reframer**| Generates dynamic FFmpeg crop-x expressions to reframe single talking-head subjects | `modules/face_tracker.py` |
| **Event Classifier** | Evaluates sustained motion, speech presence, audio transients, and scenic vistas conservatively | `modules/event_classification.py` |
| **Highlight Scorer** | Ranks candidate 30–45s windows with 10 evidence factors, opening 3s hook score, and duration fit | `modules/highlight_scoring.py` |
| **Title & Hook AI** | Interfaces with local Ollama (`llama3.2`) to generate viral titles, hooks, descriptions, and hashtags | `modules/title_generator.py`, `modules/description_generator.py` |
| **Subtitle Generator** | Generates animated ASS karaoke captions, word-level highlights, safe margins, and emojis | `modules/subtitle_generator.py`, `modules/caption_templates.py` |
| **Clip Editor** | Executes FFmpeg filtergraphs for 9:16 vertical cropping, blurred backgrounds, jump cuts, and facecam split | `modules/clip_editor.py` |
| **Thumbnail Creator** | Extracts peak high-energy frame scored by sharpness and brightness with title card | `modules/thumbnail_generator.py` |
| **Quality Control** | Inspects final exported MP4 for freeze frames, black frames, audio silence, and duration errors | `modules/quality_control.py` |
| **Metadata Manager** | Persists clip packages, export manifests, and real-world performance metrics into SQLite | `modules/metadata_manager.py` |

## Pattern Overview

**Overall:** Sequential Multi-Modal Pipeline with Isolated Resource Scheduling

**Key Characteristics:**
- **Step-Wise Phased Execution:** Pipeline strictly runs Step 1 (Ingest) → Step 2 (Audio & Speech) → Step 3 (Visual & Facecam) → Step 4 (Scoring & Metadata) → Step 5 (Cutting & Subtitling).
- **GPU VRAM Isolation:** To fit within a 6GB VRAM budget (RTX 3050 Laptop), GPU models are not held in memory simultaneously. Speech recognition unloads/releases GPU memory (`release_gpu_memory()`) before YOLO visual inference loads.
- **Explainable Multi-Factor Scoring:** Instead of black-box prediction, highlight scoring fuses bounded acoustic, visual, temporal, and narrative evidence vectors with clear factor breakdowns.

## Layers

**Ingestion Layer:**
- Location: `modules/video_ingestion.py`
- Contains: `probe_video`, `download_youtube_video`, `capture_live_stream_segment`, `accept_local_file`, `remux_to_mp4`
- Depends on: `yt-dlp`, `ffprobe`, `ffmpeg`
- Used by: `app.py`

**Signal Extraction Layer:**
- Location: `modules/audio_analysis.py`, `modules/speech_recognition.py`, `modules/object_detection.py`, `modules/facecam.py`, `modules/ocr_detection.py`
- Contains: Feature extraction over audio and visual streams at 1-second discretization
- Depends on: `librosa`, `faster-whisper`, `ultralytics` YOLO, `opencv-python`
- Used by: `app.py` (orchestrated into session state)

**Fusion & Reasoning Layer:**
- Location: `modules/event_classification.py`, `modules/highlight_scoring.py`
- Contains: Temporal window scanning, multi-factor scoring formula, opening hook analysis, facecam quota matching
- Depends on: Signal extraction arrays from previous layer
- Used by: `app.py`

**Creative & Generative Layer:**
- Location: `modules/title_generator.py`, `modules/description_generator.py`, `modules/subtitle_generator.py`, `modules/caption_templates.py`
- Contains: Subtitle layout generation (.ass / .srt), keyword emoji injection, Ollama prompt orchestration
- Depends on: `ollama`
- Used by: `app.py`, `modules/clip_editor.py`

**Rendering & Persistence Layer:**
- Location: `modules/clip_editor.py`, `modules/thumbnail_generator.py`, `modules/quality_control.py`, `modules/metadata_manager.py`
- Contains: Complex FFmpeg filtergraph construction, NVENC encoding execution, technical validation, SQLite persistence
- Depends on: `ffmpeg`, `sqlite3`
- Used by: `app.py`

## Data Flow

### Primary Request Path

1. **Video Ingestion:** User inputs URL or file (`app.py:328-612`), probe metadata extracted via `probe_video` (`modules/video_ingestion.py:18`).
2. **Audio & Speech Analysis:** `analyze_audio` computes 1-second RMS loudness and onset transients (`modules/audio_analysis.py:71`); `transcribe_audio` runs Faster-Whisper on CUDA generating dialogue and word timestamps (`modules/speech_recognition.py:46`). GPU VRAM released.
3. **Visual & Facecam Detection:** `analyze_visuals` extracts frames at 1 FPS via FFmpeg pipe and runs YOLO nano + Haar facecam tracking (`modules/object_detection.py:108`). `summarize_facecam` clusters persistent webcam locations (`modules/facecam.py:101`). GPU VRAM released.
4. **Scoring & Window Selection:** `classify_events` generates conservative event tags (`modules/event_classification.py:11`). `compute_per_second_scores` calculates 10-factor scores (`modules/highlight_scoring.py:53`). `select_highlights` runs sliding search for viral 30–45s windows with opening 3s hook scoring (`modules/highlight_scoring.py:101`).
5. **Metadata Generation:** Ollama generates 3 viral titles, 3 hooks, platform descriptions, and hashtags (`modules/title_generator.py`, `modules/description_generator.py`).
6. **Subtitles & Export:** `generate_clip_ass` builds styled ASS captions with safe-zone positioning (`modules/subtitle_generator.py:126`). `cut_clip` invokes FFmpeg to cut video, reframing into 9:16 (standard crop, blurred background, or `_cut_facecam_split`), burning subtitles with NVENC hardware acceleration (`modules/clip_editor.py:190`).
7. **Validation & Save:** `evaluate_clip` validates output integrity (`modules/quality_control.py`); `save_clip_metadata` saves records to `database/clips.db` and writes bundle to `output/{clip_id}/`.

### Streamer Reaction Facecam Path

1. During Step 3, `FacecamAnalyzer.analyze` detects edge-adjacent small faces (`w/width < 0.34`, `edge_distance < 0.34`) and computes smile/motion proxies.
2. `summarize_facecam` clusters detections into coarse 80x60 spatial bins, identifies the persistent webcam region, and calculates per-second reaction values.
3. In Step 4, `facecam_scores` are computed (`0.58 * reaction + 0.27 * gameplay + 0.15 * audio_spike`). Quota-based selection picks top reaction windows.
4. In Step 5, `build_facecam_layout` computes piecewise tracked coordinates and an inset occlusion rectangle.
5. `_cut_facecam_split` builds a complex FFmpeg filter chain that crops the facecam to the top panel, masks the original facecam in the gameplay using `delogo`, overlays gameplay on blurred gameplay in the bottom panel, inserts an optional headline banner, and burns subtitles.

## Architectural Constraints

- **Single-Threaded Streamlit Orchestration:** All processing runs synchronously inside the Streamlit script execution thread. Progress updates use callbacks to `st.progress`.
- **GPU VRAM Budget (6 GB Max):** Requires sequential model execution. `release_gpu_memory()` must be called between Faster-Whisper and YOLO.
- **Windows Path & DLL Escaping:** FFmpeg filter paths on Windows require colon and backslash escaping (`C\:/path/to/file`). CUDA DLLs must be injected using `os.add_dll_directory` for CTranslate2.
- **FFmpeg Execution Timeout:** Hard cap of 600s per clip export in `_execute_ffmpeg`.

## Anti-Patterns

### Inset In-Painting via `delogo` Smudge

**What happens:** To prevent the facecam from appearing twice in the split layout, `_cut_facecam_split` applies FFmpeg's `delogo` filter over the median webcam coordinates in the lower gameplay panel.
**Why it's wrong:** `delogo` creates an ugly, blurry smudge that ruins gameplay HUD elements, minimaps, or action.
**Do this instead:** Apply smart crop / zoom on gameplay to exclude the corner, or allow clean gameplay framing without `delogo` artifacts.

### Haar Cascade for Face Tracking & Emotion

**What happens:** `FacecamAnalyzer` relies on OpenCV Haar frontal face cascade and smile cascade on 1 FPS downsampled 640x360 frames.
**Why it's wrong:** Haar cascades fail when the streamer turns their head, laughs, looks at chat, or changes expression. The smile cascade has severe false positive rates and cannot recognize hype, shock, fear, or excitement.
**Do this instead:** Use modern lightweight neural face detectors (MediaPipe Face Landmarker or YOLO-Face) and acoustic vocal excitement models.

### Silently Disabling Dead-Air Removal on Facecam Clips

**What happens:** In `app.py:890` and `clip_editor.py:536`, `remove_dead_air` is hard-disabled if `facecam_layout` is present (`remove_dead_air and not is_facecam_clip`).
**Why it's wrong:** Streamer reaction clips retain long awkward silent pauses because jump-cuts were not reconciled with multi-stream split filtergraphs and face tracking.
**Do this instead:** Support jump-cut segment remapping across both facecam crop tracks and gameplay streams.

---

*Architecture analysis: 2026-11-18*

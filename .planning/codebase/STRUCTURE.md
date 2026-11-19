# Codebase Structure
**Analysis Date:** 2026-11-18

## Directory Layout

```
c:\ai-video-clipper\
├── .streamlit/                # Streamlit configuration (theme, upload limits)
│   └── config.toml
├── database/                  # SQLite database directory
│   └── clips.db
├── input/                     # Downloaded or user-uploaded input video files
├── output/                    # Exported clip bundles organized by clip_id
│   └── clip_01_.../           # Contains clip.mp4, thumbnail.jpg, .ass, .srt, metadata.json
├── temp/                      # Intermediate temporary scratch files (audio, temp ASS)
├── models/                    # Model weights directory (Whisper cache, custom models)
├── logs/                      # Application log files
├── downloads/                 # Legacy video download directory (symlinked/compat)
├── clips/                     # Legacy export directory (compat)
├── modules/                   # Core business logic and processing modules
│   ├── audio_analysis.py      # Librosa energy and transient analysis
│   ├── caption_templates.py   # Caption styling presets and fonts
│   ├── clip_editor.py         # FFmpeg video cutter, layouts, and NVENC export
│   ├── description_generator.py # Platform description & hashtag generator (Ollama)
│   ├── event_classification.py# Heuristic gameplay event classification
│   ├── facecam.py             # Facecam detection, clustering, reaction proxies
│   ├── face_tracker.py        # Face-following crop-x expression generator
│   ├── gpu_utils.py           # PyTorch CUDA detection & VRAM management
│   ├── highlight_scoring.py   # Highlight ranking, windowing, hook evaluation
│   ├── metadata_manager.py    # SQLite database operations and JSON exports
│   ├── object_detection.py    # YOLO nano inference and motion tracking
│   ├── ocr_detection.py       # Tesseract OCR for HUD text extraction
│   ├── quality_control.py     # Post-render technical validation
│   ├── speech_recognition.py  # Faster-Whisper transcription wrapper
│   ├── subtitle_generator.py  # ASS / SRT kinetic subtitle generation
│   ├── thumbnail_generator.py # Frame extraction and thumbnail overlay
│   ├── title_generator.py     # Title and hook generation (Ollama)
│   └── video_ingestion.py     # YouTube / Livestream / Local video ingest
├── tests/                     # Unit and integration test suites
│   ├── test_caption_templates.py # Tests for subtitle styling presets
│   ├── test_modules.py        # 9 unit tests verifying all core modules
│   ├── test_viral_features.py # Tests for viral 30-45s duration, hooks, subtitles
│   ├── stress_test_*.py       # Stress testing scripts
├── app.py                     # Streamlit frontend UI and pipeline orchestrator
├── config.py                  # Global configuration, constants, factor weights
├── requirements.txt           # Core Python dependencies
├── yolo11n.pt                 # YOLO11 nano PyTorch weights file
├── HIGHLIGHT_SELECTION_AUDIT.md # Historical audit of highlight selection logic
├── PROJECT_PROGRESS.md        # Technical documentation and implementation log
└── README.md                  # Project overview and setup instructions
```

## Directory Purposes

**`modules/`:**
- Purpose: Contains all domain-specific logic, feature extractors, and editing processors.
- Contains: Pure Python modules with minimal UI entanglement.
- Key files: `clip_editor.py`, `highlight_scoring.py`, `facecam.py`, `speech_recognition.py`.

**`tests/`:**
- Purpose: Automated test verification.
- Contains: `unittest` test classes and standalone test runner scripts.
- Key files: `test_modules.py`, `test_viral_features.py`.

**`output/`:**
- Purpose: Output directory for production-ready clip packages.
- Contains: Subfolders named `{clip_id}` each containing `clip.mp4`, `thumbnail.jpg`, subtitles, and metadata.
- Generated: Yes. Committed: No.

**`database/`:**
- Purpose: Persistent SQLite database.
- Key files: `clips.db` storing clip records, timestamps, score breakdowns, and social metrics.

## Key File Locations

**Entry Points:**
- `app.py`: Streamlit application UI and pipeline step orchestrator. Run via `.\venv\Scripts\python.exe -m streamlit run app.py`.

**Configuration:**
- `config.py`: Primary tuneable parameters (paths, VRAM caps, model sizes, factor weights, export resolutions, ASS margins).
- `.streamlit/config.toml`: Streamlit server upload file size limit (50GB).

**Core Logic:**
- `modules/highlight_scoring.py`: 10-factor multi-modal scoring and temporal window selection.
- `modules/clip_editor.py`: FFmpeg filtergraph construction, hardware NVENC encoding, and layout rendering.
- `modules/facecam.py`: Inset webcam face detection, spatial clustering, and reaction proxies.

**Testing:**
- `tests/test_modules.py`: Integration test suite covering all modules.
- `tests/test_viral_features.py`: Tests for viral duration, kinetic subtitles, safe zones, and dead-air remapping.

## Naming Conventions

**Files:**
- Snake case for Python modules: `video_ingestion.py`, `clip_editor.py`.
- Test files prefixed with `test_`: `test_modules.py`.
- Markdown documentation in uppercase: `README.md`, `PROJECT_PROGRESS.md`.

**Functions & Variables:**
- Standard Python `snake_case`: `compute_per_second_scores()`, `select_highlights()`.
- Private/internal functions prefixed with underscore: `_cut_facecam_split()`, `_use_nvenc()`.
- Global constants in `config.py` in uppercase: `EXPORT_WIDTH`, `CLIP_VIRAL_MIN_DURATION`.

## Where to Add New Code

**New Feature (e.g. New Layout / Video Filter):**
- Implementation: Add rendering logic to `modules/clip_editor.py`.
- Configuration: Add tuneable settings to `config.py`.
- UI Integration: Expose options in `app.py` under the appropriate step or sidebar.
- Tests: Add corresponding unit tests in `tests/test_viral_features.py` or new test file.

**New AI Detection Model (e.g. Facial Landmark / Emotion Detector):**
- Implementation: Create module in `modules/` (e.g. `modules/emotion_detection.py`).
- GPU Management: Ensure explicit memory release using `modules.gpu_utils.release_gpu_memory()`.
- Scoring Hook: Connect into `modules/highlight_scoring.py` factor dict.

**Utilities:**
- System/Hardware helpers: `modules/gpu_utils.py`.
- Video/Format helpers: `modules/video_ingestion.py`.

## Special Directories

**`input/`:**
- Contains downloaded source videos and uploaded files.
- Generated: Yes. Committed: No.

**`output/`:**
- Contains generated clip bundles.
- Generated: Yes. Committed: No.

**`temp/`:**
- Scratch space for temporary audio extractions, trimmed clips, and ASS files.
- Generated: Yes. Committed: No.

**`database/`:**
- Contains `clips.db`.
- Generated: Yes. Committed: No (in `.gitignore`).

---

*Structure analysis: 2026-11-18*

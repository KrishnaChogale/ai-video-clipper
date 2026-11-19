# Coding Conventions
**Analysis Date:** 2026-11-18

## Naming Patterns

**Files:**
- Snake_case for all module files: `modules/video_ingestion.py`, `modules/clip_editor.py`
- Test files prefixed with `test_`: `tests/test_modules.py`, `tests/test_viral_features.py`
- Upper-case for root project documentation: `README.md`, `PROJECT_PROGRESS.md`

**Functions:**
- Snake_case for public functions: `compute_per_second_scores()`, `select_highlights()`, `transcribe_audio()`
- Internal / private helper functions prefixed with single underscore: `_execute_ffmpeg()`, `_cut_facecam_split()`, `_tracked_crop_expression()`, `_use_nvenc()`

**Variables:**
- Snake_case for standard local and argument variables: `source_path`, `start_sec`, `end_sec`, `words_list`
- Constants in `config.py` defined in UPPER_SNAKE_CASE: `EXPORT_WIDTH`, `CLIP_VIRAL_MIN_DURATION`, `SCORE_WEIGHTS`

**Types & Classes:**
- PascalCase for classes: `FacecamAnalyzer` (`modules/facecam.py`), `TestViralFeatures` (`tests/test_viral_features.py`)
- Python 3.10+ type hinting syntax: `dict | None`, `tuple[bool, str]`, `list[dict]`

## Code Style

**Formatting:**
- PEP 8 standard formatting across all modules
- Tab / Indent: 4 spaces indentation

**Linting & Type Safety:**
- Explicit type annotations on module boundaries and public function signatures
- Defensive casting (`int(...)`, `float(...)`, `round(..., 2)`) when parsing metadata, numbers, or FFmpeg output

## Import Organization

**Order:**
1. Future imports (`from __future__ import annotations` where used)
2. Standard library modules (`os`, `sys`, `subprocess`, `json`, `logging`, `math`, `re`, `shutil`)
3. `sys.path` injection to allow direct module execution:
   ```python
   import sys, os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   import config
   ```
4. Third-party packages (`numpy`, `cv2`, `torch`, `librosa`, `faster_whisper`, `ultralytics`)
5. Local module imports (`from modules.gpu_utils import release_gpu_memory`)

**Path Aliases:**
- No package path aliases; relative path normalization and `BASE_DIR` references used consistently.

## Error Handling

**Patterns:**
- **Graceful Multi-Tier Fallbacks:** Critical operations must never crash the entire pipeline:
  - Hardware Acceleration: CUDA FP16 → CPU int8 (`modules/speech_recognition.py`)
  - Video Codec: `h264_nvenc` → `libx264` (`modules/clip_editor.py`)
  - Video Probing: `ffprobe` subprocess → OpenCV `cv2.VideoCapture` fallback (`modules/video_ingestion.py`)
  - LLM Generation: Ollama API call → Hardcoded rule-based templates if offline / timeout (`modules/title_generator.py`)
- **Explicit Exception Logging:** Catches broad exceptions defensively with `logger.warning(...)` or `logger.debug(...)` and falls back cleanly.
- **Result Tuples:** Editor and processor functions frequently return `(success: bool, result_or_error: str)` instead of raising exceptions into the UI layer.

## Logging

**Framework:**
- Standard Python `logging` module
- Module-level loggers initialized via `logger = logging.getLogger(__name__)`

**Patterns:**
- `logger.info()`: Key milestone events (e.g. model loaded, encoding starting, candidates selected)
- `logger.warning()`: Recoverable errors and fallbacks triggered (e.g. NVENC failure, CUDA memory pressure)
- `logger.debug()`: Granular audio frame loading details and silencedetect outputs

## Comments

**When to Comment:**
- Header docstrings at top of all module files explaining purpose and integration role
- Explicit rationale comments explaining why certain algorithms or heuristics were chosen (e.g., explaining why audio transients are capped or why facecam delogo coordinates are expanded)

**JSDoc/TSDoc:**
- Python standard docstrings (`"""..."""`) on all major functions detailing parameters, return types, and side effects.

## Function Design

**Size:**
- Moderate modular functions (typically 30–80 lines). Complex functions like `_cut_facecam_split` and `select_highlights` decompose sub-tasks into discrete helper blocks.

**Parameters:**
- Clear parameter names with explicit defaults sourced from `config.py`.

**Return Values:**
- Structured dictionaries or named tuples containing rich metadata (e.g., candidate dicts containing `start`, `end`, `score`, `score_factors`, `hook_score`).

## Module Design

**Exports:**
- Explicit function-based API. Minimal reliance on mutable module state (except for cached model singletons).

**Barrel Files:**
- `modules/__init__.py` exists as a package marker. Modules are imported directly by name (`from modules.clip_editor import cut_clip`).

---

*Convention analysis: 2026-11-18*

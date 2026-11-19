# Testing Patterns
**Analysis Date:** 2026-11-18

## Test Framework

**Runner:**
- Standard Python `unittest` framework (Python 3.12.10 built-in)
- Standalone execution scripts

**Assertion Library:**
- Standard `unittest.TestCase` assertions (`self.assertEqual`, `self.assertGreaterEqual`, `self.assertIn`, `self.assertAlmostEqual`)
- Python native `assert` statements in `tests/test_modules.py`

**Run Commands:**
```powershell
# Run viral features test suite
.\venv\Scripts\python.exe tests\test_viral_features.py

# Run core modules integration test suite
.\venv\Scripts\python.exe tests\test_modules.py

# Run caption templates test suite (requires PYTHONPATH set to root)
$env:PYTHONPATH="C:\ai-video-clipper"; .\venv\Scripts\python.exe tests\test_caption_templates.py
```

## Test File Organization

**Location:**
- Dedicated `tests/` directory at the project root (`tests/`)

**Naming:**
- Unit test files: `test_<feature_or_module>.py` (e.g. `test_modules.py`, `test_viral_features.py`, `test_caption_templates.py`)
- Stress and failure test files: `stress_test_<area>.py` (e.g. `stress_test_video_pipeline.py`, `stress_test_hardware.py`)

**Structure:**
```
tests/
├── test_modules.py            # Integration tests across all modules
├── test_viral_features.py     # 30-45s duration, hook scoring, dead air, subtitle remapping
├── test_caption_templates.py  # ASS style rendering for all 5 caption templates
├── stress_test_hardware.py    # VRAM stress testing
├── stress_test_video_pipeline.py # End-to-end video processing resilience
├── stress_test_inspection.py  # Output video integrity inspection
└── stress_test_failures.py    # Corrupted media and edge-case handling
```

## Test Structure

**Suite Organization:**
```python
import unittest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.highlight_scoring import select_highlights, _evaluate_hook

class TestViralFeatures(unittest.TestCase):
    def test_viral_duration_and_hook_selection(self):
        scores = [15.0] * 90
        scores[20] = 75.0
        scores[21] = 82.0
        events = [{"events": ["police_chase"], "confidence": 0.85, "category": "action"} if 20 <= i < 55 else {} for i in range(90)]
        
        hook_score, hook_type = _evaluate_hook(scores, events, 20, 55)
        self.assertGreaterEqual(hook_score, 70.0)
        self.assertEqual(hook_type, "police_gunfight")
```

**Patterns:**
- **Synthetic Data Generation:** Tests construct synthetic arrays of audio energy, visual detection scores, and transcript words without requiring gigabytes of raw gameplay video files.
- **Timestamp Arithmetic Verification:** Exact floating-point math assertions checking word shifts across dead-air cut intervals.
- **String & Tag Matching:** Regex and substring assertions checking for mandatory ASS tags (e.g., `{\c&H0000FFFF&}` active word highlight, `MarginV=440` safe margins).

## Mocking

**Framework:**
- Manual synthetic object creation (no `unittest.mock` heavy patching).

**Patterns:**
- Mocking Whisper word lists as dictionaries `[{"start": 1.0, "end": 1.4, "word": "Holy"}, ...]`.
- Mocking audio signal arrays via `np.sin` waves:
  ```python
  t = np.linspace(0, duration, sr * duration, endpoint=False)
  y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
  ```

**What to Mock:**
- Raw multi-gigabyte video stream inputs
- External Ollama local server responses
- GPU driver responses for CPU testing

**What NOT to Mock:**
- Subtitle parsing, formatting, and remapping functions
- Highlight window selection and duration fit algorithms
- Mathematical scoring equations

## Coverage

**Requirements:**
- None strictly enforced via CI gates.

**Current Test Status:**
- `test_viral_features.py`: 5/5 passed
- `test_modules.py`: 9/9 passed
- `test_caption_templates.py`: Requires `$env:PYTHONPATH="."` to resolve `config.py`

## Test Types

**Unit Tests:**
- Test individual functions: profanity censoring, ASS timestamp conversion, energy bucket generation, hook evaluation, title formatting.

**Integration Tests:**
- `tests/test_modules.py` runs tests sequentially across all modules (GPU utils, audio buckets, highlight selection, title generation, quality control, database).

**End-to-End & Stress Tests:**
- `tests/stress_test_video_pipeline.py` and `tests/stress_test_failures.py` test FFmpeg pipeline survival against truncated or corrupted files.

---

*Testing analysis: 2026-11-18*

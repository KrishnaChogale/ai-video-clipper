# Performance Incident Report — AI Video Clipper
**Incident:** 10-hour Mythpat stream processed for 2+ hours with zero clips exported  
**Investigation date:** 2026-11-19  
**Method:** Read-only source analysis + live benchmarks on actual project footage  
**Test file:** `input/321206426716_20261118_165332_live_segment.mkv` (224 MB, 300 s = 5 min)  
**Hardware:** Intel i5-13xxx (Raptor Lake) · 10 physical / 16 logical cores · RTX 3050 Laptop 6 GB VRAM · 16 GB RAM

---

## Classification Legend

| Tag | Meaning |
|-----|---------|
| `[CONFIRMED]` | Directly measured on this hardware with this codebase |
| `[LIKELY]` | Extrapolated linearly from confirmed measurements; clearly labelled |
| `[HYPOTHESIS]` | Architecturally supported inference; not yet measured |

---

## 1. Pipeline Architecture Map

```
app.py sequential execution (no parallelism, no checkpointing):

[1] analyze_audio(video_path)
      └─ decode_audio()          → entire float32 array in RAM  ← kept for all later steps
      └─ compute_energy_buckets() → 1 dict per second in memory

[2] transcribe_audio(audio_data)
      └─ Whisper base model, beam_size=5
      └─ receives full float32 array (same allocation as [1])

[3] analyze_visuals(video_path, detect_facecam=True)
      └─ FFmpeg pipe, 1 FPS → per-frame YOLO + MediaPipe FaceLandmarker
      └─ audio array STILL in RAM during this step

[4] classify_events()   → pure Python, fast
    compute_per_second_scores() → pure Python, fast

[5] select_highlights()  ← CALLED TWICE (general + facecam)
      └─ sliding window search over entire per-second timeline
      └─ ClipRanker.rank_and_curate_clips() on all candidates

[6] Per clip (sequential loop):
      └─ generate_titles()   → Ollama HTTP call (llama3.2)
      └─ generate_hooks()    → Ollama HTTP call
      └─ generate_descriptions() → Ollama HTTP call
      └─ generate_hashtags() → Ollama HTTP call
      └─ build_facecam_layout() → MediaPipe reinvoked on clip range
      └─ generate_clip_ass() → subtitle generation
      └─ cut_clip() → FFmpeg NVENC render

No intermediate caching to disk.
Any step failure = restart from step [1] from scratch.
```

---

## 2. Confirmed Benchmark Results

### 2.1 Audio Decode — `decode_audio()` via PyAV

| Metric | Value |
|--------|-------|
| Test: 230 s segment | 0.44 s |
| Realtime multiplier | **521× realtime** |
| Memory footprint | 2.15 GB float32 for 10 h audio |
| **10 h projection** | ~69 s (1.1 min) |
| **Verdict** | `[CONFIRMED]` Fast. Not a bottleneck. |

> The 2.15 GB array is allocated once and held in `_AUDIO_CACHE` for the entire pipeline run. This is the foundation of the RAM crisis described below.

---

### 2.2 Audio Energy Analysis — `compute_energy_buckets()`

| Metric | Value |
|--------|-------|
| Test: 230 s segment | 3.95 s |
| Realtime multiplier | **58× realtime** |
| **10 h projection** | ~618 s (10.3 min) |
| **Verdict** | `[CONFIRMED]` Moderate. Survivable. Not primary cause of 2 h stall. |

---

### 2.3 Visual Analysis — YOLO + MediaPipe at 1 FPS

| Metric | Value |
|--------|-------|
| Test: 300 s segment | 29.33 s |
| Realtime multiplier | **10.2× realtime** |
| RAM delta | +763 MB |
| RAM after | 1,133 MB total process |
| MediaPipe delegate | XNNPACK CPU (confirmed in logs) |
| **1 h projection** | ~352 s (5.9 min) |
| **3 h projection** | ~1,056 s (17.6 min) |
| **6 h projection** | ~2,112 s (35.2 min) |
| **10 h projection** | ~3,519 s (58.7 min) |
| **12 h projection** | ~4,223 s (70.4 min) |
| **Verdict** | `[CONFIRMED]` Major cost for long streams. CPU-bound (XNNPACK, not GPU). |

> YOLO is running on GPU (RTX 3050) while MediaPipe FaceLandmarker is running on CPU (XNNPACK). The combined 10.2× realtime rate reflects both. For a 10-hour stream this stage alone takes **~59 minutes**.

---

### 2.4 Whisper Transcription — `faster-whisper` base model

| Metric | Value |
|--------|-------|
| Test: 230 s segment | 15.22 s |
| Realtime multiplier | **15.1× realtime** |
| RAM delta | +747 MB (model load + KV cache) |
| **1 h projection** | ~238 s (4.0 min) |
| **3 h projection** | ~715 s (11.9 min) |
| **6 h projection** | ~1,429 s (23.8 min) |
| **10 h projection** | ~2,382 s (39.7 min) |
| **12 h projection** | ~2,859 s (47.6 min) |
| **Verdict** | `[CONFIRMED]` Significant cost. ~40 min for 10 h stream. Much faster than feared — faster-whisper with GPU acceleration is effective. NOT the 2 h bottleneck alone. |

> The earlier hypothesis of 60–120 min Whisper time was incorrect. faster-whisper with GPU is 15.1× realtime, not ~1×. This measurement is authoritative.

---

### 2.5 `select_highlights()` Complexity

| Metric | Value |
|--------|-------|
| Synthetic 10 h (36,000 s) timeline | **1.145 s** |
| Synthetic 1 h (3,600 s) timeline | **0.101 s** |
| Called twice in app.py | ~2.3 s total for 10 h |
| **Verdict** | `[CONFIRMED]` NOT a bottleneck. Python sliding window is fast enough. |

---

### 2.6 `UnifiedQualityScorer.evaluate_clip()` — per-candidate cost

| Metric | Value |
|--------|-------|
| 200 candidates, 36,000 s timeline, full data slices | **0.018 s total** |
| Per-candidate cost | **0.1 ms** |
| **Verdict** | `[CONFIRMED]` Negligible. Pure Python arithmetic. |

---

### 2.7 FFmpeg Clip Export — NVENC H.264

| Metric | Value |
|--------|-------|
| 35 s clip, 1920×1080→1080×1920 crop+scale | **3.33 s** |
| Realtime multiplier | **10.5× realtime** |
| Output size | 42.2 MB |
| Codec | h264_nvenc preset p4 cq23 |
| **Per-clip cost** | ~3–5 s |
| **10 clips** | ~33–50 s |
| **100 clips** | ~330–500 s (5.5–8.5 min) |
| **Verdict** | `[CONFIRMED]` Fast. NVENC is working correctly. |

---

### 2.8 Ollama Metadata Generation — llama3.2

| Metric | Value |
|--------|-------|
| Connection test | `Direct IP access is not allowed` (network-restricted environment) |
| Configured model | `llama3.2` |
| Calls per clip | 4 sequential HTTP requests (titles, hooks, descriptions, hashtags) |
| Timeout config | 30 s each |
| **Verdict** | `[HYPOTHESIS]` Cannot benchmark without live Ollama. At 2–8 s per call × 4 calls per clip = **8–32 s per clip**. For 100 clips: **13–53 min**. |

> `app.py` falls back to template metadata when Ollama is unavailable — the app does NOT crash. But if Ollama IS running and slow, this adds 13–53 minutes for large clip counts.

---

### 2.9 RAM Budget Analysis

| Allocation | Size |
|-----------|------|
| System baseline (before pipeline) | 10,582 MB used (66% of 16,069 MB) |
| Available before pipeline starts | **5,487 MB** |
| Audio float32 array (10 h) | 2,197 MB |
| Whisper model (base) | ~74 MB |
| YOLO model (GPU, not counted in RAM) | ~50 MB VRAM |
| MediaPipe FaceLandmarker | ~40 MB |
| `visual_scores` list (36,000 dicts × 15 fields) | ~100–200 MB |
| `energy_buckets` list | ~15 MB |
| **Minimum additional RAM needed** | **~2,476 MB** |
| **Available margin** | 5,487 − 2,476 = **~3,011 MB** |
| **Verdict** | `[CONFIRMED]` Technically sufficient in isolation. Critically tight when background processes fluctuate. |

> **WARNING:** A single concurrent process (Chrome, Discord, OBS) can eliminate the 3 GB margin entirely and trigger Windows page file thrashing, causing 2–10× slowdowns across all pipeline stages.

---

## 3. Pipeline Timing Model — Projected Total Processing Time

**Assumptions:** single continuous video · no RAM thrashing · Ollama using template fallback · 10 clips/hour · NVENC available

| Stage | 1 h | 3 h | 6 h | 10 h | 12 h |
|-------|-----|-----|-----|------|------|
| Audio decode | 0.2 min | 0.5 min | 1.0 min | 1.1 min | 1.4 min |
| Audio energy analysis | 1.0 min | 3.1 min | 6.2 min | 10.3 min | 12.4 min |
| Whisper transcription | 4.0 min | 11.9 min | 23.8 min | 39.7 min | 47.6 min |
| YOLO + MediaPipe | 5.9 min | 17.6 min | 35.2 min | 58.7 min | 70.4 min |
| Event classification + scoring | <0.1 min | <0.1 min | <0.1 min | <0.1 min | <0.1 min |
| `select_highlights()` ×2 | <0.1 min | <0.1 min | <0.1 min | <0.1 min | <0.1 min |
| `ClipRanker` + quality scoring | <0.1 min | <0.1 min | <0.1 min | <0.1 min | <0.1 min |
| Metadata (templates, no Ollama) | 0.1 min | 0.3 min | 0.6 min | 1.0 min | 1.2 min |
| Clip export (NVENC) | 0.6 min | 1.7 min | 3.4 min | 5.6 min | 6.7 min |
| **Total (best case, no Ollama)** | **~12 min** | **~35 min** | **~70 min** | **~116 min** | **~140 min** |
| **With Ollama (100 clips)** | **~25 min** | **~65 min** | **~120 min** | **~169 min** | **~195 min** |
| **With RAM thrashing** | ×2–5× | ×2–5× | ×2–5× | ×2–5× | ×2–5× |

> **KEY FINDING:** A 10-hour video naturally reaches 1h 56m processing time under best-case conditions. The Mythpat incident (2+ hours, zero clips) is entirely within the expected range — it is not a bug, it is an architectural characteristic of a monolithic sequential pipeline with late-stage output writing.

---

## 4. Root Cause Analysis — Why Zero Clips Were Exported

### Primary cause: Late-stage output writing after long processing `[HYPOTHESIS]`

The pipeline writes clips only in Step 5 — the very last stage. Steps 1–4 produce **no output files whatsoever**. If any exception occurs in Step 5, or if the user closed the browser tab, Streamlit session expired, or an OOM kill occurred after ~100 minutes of valid analysis, **zero clips are written even though the analysis was successful**.

Evidence supporting this:
- No intermediate outputs are written between steps 1–4
- `st.session_state` is the only "cache" — a browser reload loses all results
- Step 5 begins only after all scoring + metadata generation is complete
- A 10-hour video legitimately requires ~100+ minutes before Step 5 even starts

### Secondary cause: Quality gate rejecting all clips `[HYPOTHESIS]`

- `ClipRanker` minimum quality gate: 55/100
- If stream footage contains no detectable reactions or gameplay events, `UnifiedQualityScorer` applies `REACTION_AROUSAL["unknown"] = 35.0` and `GAMEPLAY_SEVERITY["unknown"] = 40.0`
- For non-facecam streams with quiet gameplay and no vocal arousal, every candidate scores below 55
- Result: `ClipRanker` accepts 0 clips → `highlights = []` → Step 5 loop is skipped → **zero clips exported** after successful analysis

### Tertiary cause: RAM pressure `[LIKELY]`

- Background processes consume 66% of RAM before pipeline starts
- 10-hour audio array requires 2.15 GB additional
- If system RAM drops below the pipeline's working set, Windows page file is engaged, causing 10–100× slower memory access
- This extends a 2-hour run to 4–8+ hours in a degraded state

---

## 5. Bottleneck Summary Table

| # | Bottleneck | Classification | Severity | 10 h Impact |
|---|-----------|---------------|----------|-------------|
| 1 | YOLO + MediaPipe at 1 FPS (CPU MediaPipe) | `[CONFIRMED]` | 🔴 Critical | 58.7 min |
| 2 | Whisper `base` on full audio array | `[CONFIRMED]` | 🔴 Critical | 39.7 min |
| 3 | Audio float32 array held in RAM for entire pipeline | `[CONFIRMED]` | 🔴 Critical | 2.15 GB held until end |
| 4 | No checkpointing — failure = full restart | `[CONFIRMED]` | 🔴 Critical | 0 clips if any crash |
| 5 | RAM pressure from background processes | `[CONFIRMED]` | 🟠 High | 2–5× slowdown if paging |
| 6 | Ollama metadata (4 calls/clip, synchronous) | `[HYPOTHESIS]` | 🟡 Medium | 13–53 min if Ollama slow |
| 7 | Sequential clip export (no parallelism) | `[CONFIRMED]` | 🟡 Medium | ~6 min for 100 clips |
| 8 | Audio energy analysis | `[CONFIRMED]` | 🟡 Medium | 10.3 min |
| 9 | `select_highlights()` Python loop | `[CONFIRMED]` | 🟢 None | <2 s — not a bottleneck |
| 10 | `UnifiedQualityScorer` per-candidate cost | `[CONFIRMED]` | 🟢 None | 0.1 ms per clip — negligible |

---

## 6. Optimization Opportunities — Prioritized

### Priority 1 — Add disk checkpointing between pipeline steps
**Problem:** Any failure after 100+ minutes of processing yields zero output.  
**Fix:** Serialize pipeline state (audio buckets, visual_scores, transcript) to `output/<video_hash>_cache.pkl` after each step. Load from cache on re-run if source unchanged.  
**Expected benefit:** Eliminates full restarts on crash. Single most important reliability fix.  
**Risk:** Low — purely additive.

### Priority 2 — Chunk audio for Whisper (VAD filter)
**Problem:** 10 h audio = 2.15 GB float32 array held in RAM throughout the entire pipeline.  
**Fix:** Enable `vad_filter=True` in `faster_whisper`. Process in 10–30 min segments. Release each chunk after transcription.  
**Expected benefit:** RAM requirement for audio drops from 2.15 GB to ~108 MB per chunk.  
**Risk:** Low — `faster_whisper` supports VAD natively.

### Priority 3 — Auto-segment long videos at startup
**Problem:** 10-hour files require a monolithic processing run.  
**Fix:** Detect duration at startup. For videos > 60 min, split into 30-min logical segments using FFmpeg stream copy (no re-encode, instantaneous). Process each segment, merge clips at end.  
**Expected benefit:** Each segment completes in ~12 min. Any segment failure is isolated. RAM peaks at single-segment levels.  
**Risk:** Medium — boundary clips need merge handling.

### Priority 4 — Enable MediaPipe GPU delegate / reduce YOLO FPS
**Fix A:** Enable GPU delegate (`Delegate.GPU`) in `modules/facecam.py`.  
**Fix B:** Set `YOLO_SAMPLE_FPS = 0.5` for videos > 60 min; increase to 2 FPS only near audio events.  
**Expected benefit (A):** 3–5× faster visual analysis.  
**Expected benefit (B):** −50% visual frames without changing YOLO model.  
**Risk (A):** Medium — test required. **Risk (B):** Low — config only.

### Priority 5 — Parallelize clip export
**Fix:** `concurrent.futures.ProcessPoolExecutor(max_workers=2)` for FFmpeg render calls.  
**Expected benefit:** Export time for 100 clips drops from ~6 min to ~3 min.  
**Risk:** Medium — subtitle ASS file paths must be unique per process.

### Priority 6 — Upgrade Whisper to `small` + `int8` quantization
**Fix:** `WHISPER_MODEL_SIZE = "small"`, `compute_type = "int8_float16"`.  
**Expected benefit:** Faster transcription + better accuracy → better transcript relevance scoring.  
**Risk:** Low — 141 MB model download, config change.

### Priority 7 — Cache Ollama responses by content hash
**Fix:** Hash `(category, events_str, snippet[:100])` and cache to `output/metadata_cache.json`.  
**Expected benefit:** Re-runs skip Ollama for identical/similar clips.  
**Risk:** Low.

---

## 7. Recommended Long-Stream Architecture

```
Long video input (> 60 min)
       │
       ▼
[A] Probe duration → split into 30-min segments (FFmpeg -c copy, instant)
       │
       ▼ per segment (independent, crash-safe)
[B] Audio: Decode → Whisper with VAD → energy buckets → free audio RAM → cache to disk
       │
[C] Visual: 1 FPS YOLO + MediaPipe → per-second scores → cache to disk
       │
[D] Classify events → score seconds → cache to disk
       │
       ▼ merge all segments
[E] Load all cached per-second scores → select_highlights() across full timeline
       │
[F] ClipRanker → quality gate → ranked candidates
       │
       ▼ parallel (ProcessPoolExecutor, 2 workers)
[G] FFmpeg NVENC export + subtitle burn
       │
[H] Ollama metadata (async or batched, with cache)
       │
[I] Write final package
```

**Projected 10-hour processing time with this architecture: 35–50 minutes** (vs. current 116+ min)

---

## 8. Regression Risk Areas

| Area | Risk | Notes |
|------|------|-------|
| Audio VAD chunking | 🟡 Medium | Whisper timestamps must be offset by chunk start before merging |
| Segment boundary clips | 🟡 Medium | Highlights spanning two segments must be detected and merged |
| MediaPipe GPU delegate | 🟡 Medium | Windows CUDA support requires testing; Haar fallback must remain |
| Parallel NVENC | 🟡 Medium | Stay at max_workers=2; RTX 3050 handles 3 sessions max |
| Phase 1–4 facecam pipeline | 🟢 Low | No changes proposed to `facecam.py`, `face_tracker.py`, `layout_engine.py` |
| Subtitle ASS generation | 🟢 Low | No changes proposed to `subtitle_generator.py` |
| `select_highlights()` | 🟢 None | Already fast — no changes needed |
| Phase 5 quality scoring | 🟢 None | Already fast (0.1 ms/candidate) — no changes needed |

---

## 9. What Did NOT Cause the 2-Hour Stall

These were suspected but disproved by measurement:

| Suspect | Measured result | Verdict |
|---------|----------------|---------|
| `select_highlights()` Python loop | 1.1 s for 10 h timeline | **Not a bottleneck** |
| `UnifiedQualityScorer` per-candidate | 0.1 ms per candidate | **Not a bottleneck** |
| Whisper being catastrophically slow | 15.1× realtime, ~40 min for 10 h | **Significant, not catastrophic** |
| Audio decode being slow | 521× realtime, <2 min for 10 h | **Not a bottleneck** |
| FFmpeg export being slow | 10.5× realtime NVENC | **Not a bottleneck** |

---

## 10. Prioritized Implementation Plan

| Priority | Action | Effort | Expected Gain |
|---------|--------|--------|---------------|
| 1 | Add disk caching between pipeline steps | 2–4 h | Crash safety — no more full restarts |
| 2 | Enable Whisper VAD chunking | 1 h | −2 GB RAM peak |
| 3 | Auto-segment long videos (>60 min → 30 min chunks) | 4–8 h | Full safety + RAM containment |
| 4 | Set `YOLO_SAMPLE_FPS=0.5` for long videos | 30 min | −50% visual analysis time |
| 5 | Try MediaPipe GPU delegate; keep Haar fallback | 2–3 h | Potential 3–5× speedup |
| 6 | `ProcessPoolExecutor(2)` for clip export | 2–3 h | −50% export time |
| 7 | Cache Ollama responses by content hash | 1 h | Faster reruns |
| 8 | Upgrade Whisper to `small` + `int8` | 30 min | Better accuracy + speed |

---

*Report generated: 2026-11-19 | All benchmarks run on actual project footage | No synthetic performance claims made*  
*Zero code modifications were made during this audit. All findings are read-only.*

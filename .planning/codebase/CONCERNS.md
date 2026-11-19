# Codebase Concerns
**Analysis Date:** 2026-11-18

## Tech Debt

**Bypassed Dead-Air Removal on Facecam Clips:**
- Issue: In `app.py` line 890 (`remove_dead_air and not is_facecam_clip`) and `clip_editor.py` line 536 (`if remove_dead_air and not facecam_layout:`), dead-air jump cuts are completely disabled for facecam reaction clips.
- Files: `app.py`, `modules/clip_editor.py`
- Impact: Streamer reaction clips retain long awkward silent pauses, reducing retention on TikTok/Shorts/Reels compared to standard gameplay clips.
- Fix approach: Implement multi-segment coordinate track slicing in `_cut_facecam_split` so jump-cuts can be applied to both facecam and gameplay streams simultaneously with synchronized subtitle remapping.

**`delogo` Filter Inset Masking Smudge:**
- Issue: To prevent the original inset facecam from duplicating in the bottom gameplay panel, `_cut_facecam_split` runs `delogo=x={overlay_x}:y={overlay_y}:w={overlay_w}:h={overlay_h}:show=0`.
- Files: `modules/clip_editor.py` (lines 429-434)
- Impact: Produces an ugly, blurry rectangular smudge artifact over the bottom panel. If the streamer's webcam overlays gameplay minimaps, weapons, or kill feeds, the HUD is visually corrupted.
- Fix approach: Implement intelligent gameplay cropping (e.g. crop the remaining 80% of the gameplay screen that excludes the webcam corner), or use edge feathering and clean overlays instead of FFmpeg's destructive `delogo`.

**Missing `sys.path` in `tests/test_caption_templates.py`:**
- Issue: `tests/test_caption_templates.py` imports `config` directly without ensuring the repository root is on `sys.path`.
- Files: `tests/test_caption_templates.py`
- Impact: Running `python tests/test_caption_templates.py` directly from PowerShell fails with `ModuleNotFoundError: No module named 'config'`.
- Fix approach: Add `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` at the top of the test file.

## Known Bugs & Fragile Areas

**Haar Cascade Head-Pose Fragility in `FacecamAnalyzer`:**
- Symptoms: Facecam detection drops completely if the streamer tilts their head, turns toward chat, looks down at their keyboard, or laughs with head thrown back.
- Files: `modules/facecam.py` (lines 15-99), `modules/face_tracker.py`
- Trigger: Non-frontal head poses (yaw/pitch > 20° drops Haar cascade detection recall to near zero).
- Workaround: System falls back to centered crop or drops the facecam quota, logging "lost face tracking before render; skipping it instead of exporting a false facecam layout".
- Fix approach: Replace OpenCV Haar Cascades with modern lightweight neural face detection (e.g., MediaPipe Face Landmarker or YOLOv8-Face nano).

**Single Static Cluster Assumption for Webcam Location:**
- Symptoms: If a streamer changes their webcam position or toggles full-screen webcam during lobby/talking scenes, the spatial clustering in `summarize_facecam` picks only the single most frequent 80x60 bin and fails on the other scenes.
- Files: `modules/facecam.py` (lines 101-163)
- Trigger: Dynamic OBS scenes (e.g., webcam moves from top-left during gameplay to bottom-right during menus, or switches between webcam and full-screen).
- Fix approach: Implement dynamic per-scene facecam tracking with temporal state transitions rather than a single static video-wide median bounding box.

## Security Considerations

**Unauthenticated Local Streamlit Deployment:**
- Risk: Streamlit server binds by default without user authentication. Anyone with network access to the port (8501) can trigger long-running GPU video processing jobs or inspect local files.
- Files: `app.py`, `.streamlit/config.toml`
- Current mitigation: Designed for single-user local workstation use (`localhost`).
- Recommendations: If exposed across a local network or VPN, enable Streamlit basic authentication or run behind a reverse proxy with auth.

**YouTube Cookie Exposure:**
- Risk: Session cookies uploaded via `uploaded_cookies.txt` or `ai-video-clippercookies.txt.txt` contain live Google/YouTube session credentials.
- Files: `uploaded_cookies.txt`, `ai-video-clippercookies.txt.txt`
- Current mitigation: Added to `.gitignore`.
- Recommendations: Ensure cookie files are never committed, and automatically purge uploaded cookie files from disk after processing.

## Performance Bottlenecks

**CPU Video Decoding During Complex Split Compositing:**
- Problem: Rendering 1080x1920 facecam split clips (`_cut_facecam_split`) decodes the input file once and splits into 4 sub-streams (face_src, game_bg, game_fg, overlay) with scaling, boxblur, and delogo.
- Files: `modules/clip_editor.py` (lines 395-482)
- Cause: FFmpeg is invoked without hardware-accelerated decode flags (`-hwaccel cuda`), meaning decoding and filtering run entirely on CPU threads before passing frames to `h264_nvenc` for encoding.
- Improvement path: Benchmark and add `-hwaccel cuda -hwaccel_output_format cuda` or optimize the filtergraph to eliminate redundant `split` and `scale` operations.

**Synchronous UI Execution in Streamlit:**
- Problem: The entire 5-step pipeline executes inside the Streamlit web thread. A browser refresh or network hiccup can interrupt execution or leave partial scratch files in `temp/`.
- Files: `app.py`
- Improvement path: Decouple execution into a background task queue (e.g., using Python `concurrent.futures` or background worker) with persistent state stored in SQLite.

## Reaction & Emotion Proxy Accuracy Limitations

**Crude Motion and Smile Proxies:**
- Problem: Reaction scoring in `modules/facecam.py` uses `0.78 * motion + 0.22 * smile`.
  - Motion is simply pixel difference `cv2.absdiff` between consecutive 48x48 face crops. It triggers on hand movements, mic adjustments, head shaking, chair movement, or lighting changes.
  - Smile is estimated using `haarcascade_smile.xml` on the lower face, which has high false-positive rates on open mouths (screaming, shouting, yawning) and fails completely on subtle smirks or micro-expressions.
- Files: `modules/facecam.py` (lines 79-92)
- Improvement path: Incorporate facial action units (AU) or a dedicated lightweight expression model (such as FER / HSEmotion) to measure valence and arousal accurately.

**Lack of Acoustic Emotion Disentanglement:**
- Problem: Audio analysis (`modules/audio_analysis.py`) relies exclusively on RMS loudness and onset transients. It cannot distinguish between the streamer screaming in joy/rage versus loud in-game engine noise, explosions, or background music.
- Files: `modules/audio_analysis.py` (lines 33-68)
- Improvement path: Integrate an audio classification model (like AST or YAMNet) or voice activity detection with pitch/spectral flux to specifically isolate human vocal excitement from game sound effects.

## Audio-Visual Alignment Limitations

**1-Second Quantization Granularity:**
- Problem: Audio energy, onset spikes, YOLO visual detections, and facecam presence are sampled and stored in integer 1-second buckets (`second = 0, 1, 2...`).
- Files: `modules/audio_analysis.py`, `modules/object_detection.py`, `modules/highlight_scoring.py`
- Impact: Sub-second comedic timing, instant headshot reactions, and precise hook triggers are quantized to coarse 1-second boundaries. A reaction occurring 300ms after a gameplay event cannot be temporally aligned with sub-second precision during scoring.
- Improvement path: Transition to 0.25s (250ms) or 0.1s temporal granularity for onset detection and visual tracking.

## Vertical Video Rendering Limitations

**Rigid Panel Aspect Ratio & Headroom Truncation:**
- Problem: `modules/facecam.py` fixes the top facecam crop aspect ratio to `panel_aspect = 1080.0 / 864.0 = 1.25` and scales height by `2.8 * face_height`.
- Files: `modules/facecam.py` (lines 136-144), `modules/clip_editor.py` (lines 406-413)
- Impact: For wide 16:9 streamer webcams, this tight crop cuts off the streamer's arms, gaming chair, and room context; for tall 4:3 webcams, it can truncate foreheads or shoulders.
- Improvement path: Allow adaptive panel aspect ratios based on detected webcam bounding box aspect ratio, with configurable headroom padding.

**Visual Clutter in Bottom Gameplay Panel:**
- Problem: Compressing 16:9 gameplay into the bottom 45–55% of a 1080x1920 canvas occupies roughly 1080x607.
- Files: `modules/clip_editor.py` (lines 425-440), `modules/subtitle_generator.py` (lines 135-150)
- Impact: When subtitles are placed at the default safe margin (`MarginV=440`), they overlap directly across the center of the compressed gameplay action, while the headline banner occupies the middle divider, leaving little unobstructed gameplay visible.
- Improvement path: Add an option to crop gameplay dynamically to 1:1 or 4:5, or position subtitles over the lower blurred pillarbox region below the gameplay feed.

## Test Coverage Gaps

**Zero Test Coverage for Facecam Modules:**
- What's not tested: `modules/facecam.py`, `modules/face_tracker.py`, and `modules/clip_editor._cut_facecam_split` have zero automated unit or integration tests in `tests/`.
- Files: `modules/facecam.py`, `modules/face_tracker.py`, `modules/clip_editor.py`
- Risk: Any refactor or update to OpenCV versions, FFmpeg filter syntax, or layout math can break facecam rendering unnoticed.
- Priority: High.

---

*Concerns audit: 2026-11-18*

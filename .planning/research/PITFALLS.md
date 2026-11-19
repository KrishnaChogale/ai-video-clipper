# Domain Pitfalls: Streamer Reaction AI Video Clipping System

**Domain:** Local Multi-Modal Streamer Reaction Video Automation  
**Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), Windows 11  
**Researched:** 2026-11-19  
**Status:** Validated  

---

## Critical Pitfalls (Causes Crashes, Rewrites, or Broken Output)

### Pitfall 1: The 6GB VRAM OOM Avalanche

**What goes wrong:** The application crashes with `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory` during video analysis or rendering.  
**Why it happens:** By default, Ollama keeps the `llama3.2` model cached in GPU VRAM for 5 minutes (`keep_alive=5m`), consuming ~2.5GB VRAM. If `faster-whisper` (1.1GB–2.5GB) is loaded concurrently, or if PyTorch allocates its caching pool (1.0GB), and then FFmpeg NVENC attempts to allocate surface buffers (~400MB), the total memory request exceeds 6,144 MiB. Windows Desktop Window Manager (DWM) also locks ~800MB VRAM, leaving only ~5.3GB usable.  
**Consequences:** Hard crash of the Streamlit application or silent termination of the video pipeline midway through a multi-hour stream.  
**Prevention:**
1. Execute models sequentially in isolated pipeline stages.
2. Purge Ollama immediately using `keep_alive=0` via the Ollama REST API or Python client.
3. Call `gc.collect()` and `torch.cuda.empty_cache()` immediately after Whisper transcription finishes and unload the model instance.
4. Run MediaPipe, HSEmotion, and YAMNet strictly on **CPU / ONNX-CPU** runtime (they execute in <15ms without needing GPU VRAM).  
**Detection:** Monitor VRAM headroom using `torch.cuda.mem_get_info()` or `nvidia-smi` before initializing each model. If available VRAM is < 1.5GB, force memory eviction.

---

### Pitfall 2: Causality Inversion & Missing Payoff (The Reaction Slicing Trap)

**What goes wrong:** The generated clip shows the streamer screaming or laughing, but cuts in *after* the in-game event already happened, or cuts off abruptly before the streamer finishes their sentence or comedic punchline.  
**Why it happens:** Human reaction latency is 300ms to 1,200ms. If an algorithm simply takes a 30-second window centered around the loudest audio spike ($T_{reaction} \pm 15s$), or starts the clip at the onset of laughter, the in-game trigger (e.g. car stunt, sniper shot, sudden jumpscare) occurred *before* the reaction onset and gets truncated. Furthermore, streamers often deliver the funniest punchline or recap 3–5 seconds *after* the initial scream.  
**Consequences:** Viewers on TikTok/Reels scroll away immediately because they have no idea *why* the streamer is laughing or screaming (missing context).  
**Prevention:**
1. Implement **Causal Reaction-Gameplay Binding**: detect the in-game trigger $T_{game}$ and the reaction climax $T_{reaction}$ as an asymmetric pair.
2. Structure the clip boundary as:
   - Lead-in / Setup: $T_{game} - 2.5\text{s}$ (shows the setup and tension building).
   - Reaction Peak: $T_{reaction}$ (viewer sees the streamer lose their mind).
   - Recovery & Punchline: $T_{reaction} + 4.0\text{s}$ (captures the post-reaction commentary).  
**Detection:** Verify that every candidate clip window has an in-game action trigger at least 1.5s into the clip, rather than in the first 0.2s or before the start.

---

### Pitfall 3: Jerky Facecam Tracking ("The Bobblehead Effect")

**What goes wrong:** The facecam panel in the vertical video visibly shakes, wobbles, or vibrates frame-by-frame, creating a distracting and nauseating visual experience.  
**Why it happens:** Raw face bounding boxes from detectors fluctuate by a few pixels every frame due to minor mouth movements, head gestures, lighting flickers, and detector noise. Direct cropping to raw detector coordinates translates this pixel noise into whole-screen camera vibration.  
**Consequences:** Low visual quality, amateur aesthetic, high viewer drop-off.  
**Prevention:**
1. Apply an **Exponential Moving Average (EMA)** filter to crop center $(c_x, c_y)$ and dimensions $(w, h)$ with smoothing factor $\alpha \approx 0.15 - 0.20$.
2. Implement a **velocity deadband**: if the face moves less than 12 pixels, freeze the crop center completely.
3. Implement a **saccade override**: if the streamer leans heavily across the frame (>120 pixels displacement), bypass smoothing and snap quickly to prevent slow, awkward panning.
4. Add fixed 20% head-room padding above the forehead and 35% shoulder-room padding below the chin.  
**Detection:** Calculate the standard deviation of crop center acceleration between consecutive seconds. It should be near zero during stable sitting.

---

### Pitfall 4: FFmpeg Complex Filtergraph Audio/Video Desync

**What goes wrong:** In jump-cut or split-screen clips, speech audio gradually falls out of sync with streamer lip movements by 500ms to 2000ms.  
**Why it happens:** When concatenating segments (`concat` filter) or applying multiple `trim` and `setpts` operations in FFmpeg without corresponding `atrim` and `asetpts=PTS-STARTPTS`, audio and video presentation timestamps (PTS) drift apart. Furthermore, variable framerate (VFR) inputs from game recording software (like OBS or shadowplay) compound timestamp drift when filtered.  
**Consequences:** Lip-sync is ruined; viewers notice words don't match mouth movements.  
**Prevention:**
1. Always pair `trim=start=X:end=Y,setpts=PTS-STARTPTS` with `atrim=start=X:end=Y,asetpts=PTS-STARTPTS` on every sub-stream.
2. Normalize inputs to constant framerate (`fps=30` or `fps=60`) and force audio resampling (`aresample=async=1000`) in the filtergraph.
3. Test every filter output using `-vsync cfr`.  
**Detection:** Check PTS delta between video and audio streams using `ffprobe -show_entries packet=pts_time,stream_index`.

---

## Moderate Pitfalls

### Pitfall 5: In-Game Sound vs Streamer Voice Confusion

**What goes wrong:** The clipper detects loud sound spikes from in-game machine guns, grenades, or car engines and flags them as "streamer screaming", creating boring clips of normal gameplay where the streamer was actually silent.  
**Why it happens:** Relying solely on raw RMS energy or decibel levels cannot distinguish between a game sound effect and a human vocal reaction.  
**Consequences:** Clips selected with high "audio excitement" have zero vocal reaction.  
**Prevention:**
1. Combine acoustic features: Human screaming or shouting exhibits high fundamental frequency shifts ($F_0 > 300\text{Hz}$) and extreme pitch acceleration, whereas engine noise is low-frequency harmonic rumble.
2. Run **YAMNet-ONNX**: It has separate class heads for `Screaming`, `Laughter`, and `Gasp` vs `Gunshot`, `Explosion`, `Vehicle`, and `Music`.
3. Weight vocal event classification higher than raw loudness.

---

### Pitfall 6: Generic Hallucinated LLM Hooks

**What goes wrong:** The generated hook headline says *"THEY DID NOT EXPECT THIS IN GTA 6 😱"* or invents details about cars and characters that never appeared in the video clip.  
**Why it happens:** The prompt given to the LLM provides generic instructions without grounding facts, or the LLM temperature is set too high ($>0.8$). Small local models (3B) easily hallucinate generic clickbait phrases if not constrained.  
**Consequences:** Clickbait that doesn't match the video content, annoying viewers.  
**Prevention:**
1. Pass exact factual anchors in the prompt: dominant detected emotion (e.g., `EMOTION: Extreme Surprise / Jaw Drop`), verified detected objects (e.g., `OBJECTS: Police Helicopter, Sports Car`), and verbatim speech transcript snippet.
2. Enforce strict JSON schema validation via Pydantic.
3. Set LLM temperature to $0.4 - 0.5$ for tighter factual grounding.

---

### Pitfall 7: "Ghost Streamer" Inset Duplication

**What goes wrong:** The streamer's facecam appears in the top split panel, but also remains visible in the corner of the bottom gameplay panel, looking awkward and cluttered.  
**Why it happens:** The source video had the streamer's webcam overlaid directly on the game stream. When the game stream is cropped for the lower panel, the original facecam inset is still there.  
**Consequences:** Unprofessional layout; two instances of the streamer on screen at once.  
**Prevention:**
1. Use the persistent facecam bounding box detected in Phase 1.
2. Apply FFmpeg `delogo` filter (`delogo=x=X:y=Y:w=W:h=H:show=0`) on the gameplay source to remove or blur out the inset.
3. If the gameplay aspect ratio leaves room, position the gameplay crop such that the original inset falls outside the visible 9:16 viewport.

---

## Minor Pitfalls

### Pitfall 8: Windows Path Escaping in FFmpeg

**What goes wrong:** FFmpeg crashes with `Cannot find fontfile` or syntax errors when burning ASS subtitles or text overlays.  
**Why it happens:** Windows paths with backslashes (`C:\ai-video-clipper\...`) and colons (`C:`) break FFmpeg filter argument parsing, where `:` is a filter option delimiter and `\` is an escape character.  
**Prevention:** Convert all file paths passed to FFmpeg filters to forward slashes and escape colons: `path.replace("\\", "/").replace(":", "\\:")`.

### Pitfall 9: Dead Air Over-Trimming

**What goes wrong:** The video cuts frantically every time the streamer takes a breath, making speech sound unnatural and frantic.  
**Prevention:** Enforce a minimum silence duration threshold ($\ge 1.0\text{s}$), add safety head/tail padding ($\ge 250\text{ms}$), and limit dead-air cuts to at most 2 per 30-second window.

---

## Phase-Specific Warnings & Mitigation Matrix

| Phase Topic | Likely Pitfall | Concrete Mitigation |
| :--- | :--- | :--- |
| **Facial Emotion** | MediaPipe fails on tilted or small inset faces | Scale and crop facecam candidate region before feeding to Landmarker; check confidence scores. |
| **Vocal Arousal** | Audio clipping/distortion confuses pitch estimator | Normalize audio volume (-20 LUFS) prior to running Librosa $F_0$ or YAMNet. |
| **Temporal Alignment** | Trigger and reaction peaks match by coincidence | Enforce strict causality constraint: $0.3\text{s} \le (T_{reaction} - T_{trigger}) \le 1.8\text{s}$. |
| **Split-Screen Layout** | Video scaling distortion / non-even dimensions | Ensure all scaled widths and heights are even numbers (`-2` or `trunc(x/2)*2`). |
| **Ollama Hook Gen** | Model stays resident in VRAM and blocks NVENC | Call `keep_alive=0` immediately upon receiving LLM response. |

---

## Sources

- **FFmpeg Bug Tracker & Filtering Documentation:** [trac.ffmpeg.org](https://trac.ffmpeg.org)
- **PyTorch CUDA Caching Allocator Docs:** [pytorch.org/docs/stable/notes/cuda.html](https://pytorch.org/docs/stable/notes/cuda.html)
- **Google MediaPipe Known Issues:** [github.com/google-ai-edge/mediapipe/issues](https://github.com/google-ai-edge/mediapipe/issues)

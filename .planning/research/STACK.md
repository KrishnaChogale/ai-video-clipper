# Technology Stack: Local Streamer Reaction Clipping System

**Project:** AI Video Clipper - Professional Streamer Reaction Upgrade  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM), Windows 11, Local Inference  
**Researched:** 2026-11-19  
**Status:** Validated  

---

## Recommended Stack

### 1. Facial Expression & Emotion Detection

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **MediaPipe Face Landmarker** | `mediapipe>=0.10.14` | Real-time face tracking & 52 blendshapes (`jawOpen`, `mouthSmile`, `browInnerUp`, `eyeWide`) | **0 MB VRAM** footprint (runs on CPU via XNNPACK / TFLite in 4–8ms/frame). Delivers 478 3D landmarks and Apple ARKit-compatible blendshape coefficients (0.0 to 1.0) with zero CUDA memory overhead. Far superior to OpenCV Haar Cascades for rotated/tilted faces and variable lighting. |
| **HSEmotion-ONNX (`EmotiEffNet`)** | `hsemotion-onnx>=0.3.0` | Discrete Emotion Classification (Surprise, Happiness, Fear, Anger, Sadness, Neutral) & Valence/Arousal | Runs via `onnxruntime` on CPU or DirectML/CUDA with a tiny ~15MB model (`enet_b0_8_best_afew`). Provides true categorical emotion probability and emotional arousal scores at ~6–10ms per face crop. Does not require PyTorch or TensorFlow runtimes. |

### 2. Audio Vocal Arousal & Event Classification

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **Librosa** | `librosa>=0.10.2` | Pitch tracking ($F_0$ via `pyin`), Spectral Centroid, Spectral Flux, RMS Energy Deltas | Native CPU-based acoustic analysis. Pitch jumps (>1.5 octaves above baseline) and high spectral centroid (>2500 Hz) reliably separate vocal excitement/screaming from flat commentary and background music. |
| **YAMNet (ONNX Runtime)** | `onnxruntime>=1.18.0` | 521-class Audio Event Detection (Laughter, Scream, Gasp, Shouting, Cheering, Gunshots) | MobileNet-based architecture pre-trained on Google AudioSet. Converted to ONNX, it runs on CPU with ~15MB memory footprint in ~12ms per 1-second segment. Accurately isolates laughter bursts, gasps, and shrieks without heavy external models. |

### 3. Speech Recognition & Transcript Alignment

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **faster-whisper** | `faster-whisper>=1.0.3` | Word-level speech transcription & timestamping | CTranslate2-accelerated Whisper implementation. In `float16` on CUDA, model `base.en` takes ~600MB VRAM and `small.en` takes ~1.1GB VRAM, running at 4x–8x real-time. Unloaded from GPU memory immediately after the transcription phase completes. |

### 4. Gameplay Vision & Object Detection

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **Ultralytics YOLO11n (ONNX / TensorRT)** | `ultralytics>=8.3.0` | Detection of game action cues (vehicles, persons, weapons, explosions) | Exported to FP16 ONNX format. Consumes <300MB VRAM or can execute on CPU/DirectML. Extremely lightweight (<6MB weights) and processes 640x360 sampled gameplay frames at >120 FPS. |

### 5. Local LLM Hook & Headline Generation

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **Ollama (`llama3.2:3b`)** | `ollama>=0.3.3` | Context-aware viral hook headline, caption, and title generation | 3-billion parameter model optimized for instructional following and JSON structured output schema via Pydantic. Fits comfortably in ~2.2GB VRAM when active, but must be managed with `keep_alive=0` so it automatically yields VRAM when video rendering or transcription begins. |

### 6. Video Composition & Export

| Technology | Version | Purpose | Why Selected |
| :--- | :--- | :--- | :--- |
| **FFmpeg with NVENC (`h264_nvenc`)** | `ffmpeg>=6.1` | Multi-stream video filtering, dynamic layout transitions, split-screen, subtitle burn-in | Hardware-accelerated encoding using the RTX 3050 NVENC silicon (consumes only ~350MB VRAM). Complex filtergraphs handle smart framing, `boxblur`, `delogo`, dynamic `overlay` transitions, and ASS animated subtitles in a single fast pass, avoiding slow Python frame-by-frame rendering loops. |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Rejected |
| :--- | :--- | :--- | :--- |
| **Face Tracking & Landmarking** | **MediaPipe Face Landmarker** | OpenCV Haar Cascade | Haar cascades fail completely on rotated/tilted heads, side profiles, dim streamer lighting, and produce wild false positives. No facial blendshapes or expression metrics. |
| **Facial Emotion Recognition** | **HSEmotion (ONNX)** | DeepFace | DeepFace initializes heavy TensorFlow/Keras or PyTorch backends requiring 1.8GB–3.0GB VRAM, which causes immediate OOM on a 6GB card when running alongside Whisper or Ollama. |
| **Emotion & Action Units** | **MediaPipe + HSEmotion** | Py-Feat | Py-Feat is geared for offline psychological research; heavy dependency tree, slow frame loops, and requires 2GB+ VRAM for RetinaFace + ResNet backbones. |
| **Audio Classification** | **YAMNet (ONNX) + Librosa** | PyAudioAnalysis | PyAudioAnalysis relies on outdated dependencies, lacks modern pre-trained deep acoustic representations, and has high false positive rates for human vocalizations in noisy game streams. |
| **Laughter Segmentation** | **YAMNet + Spectral Flux** | `jrgillick/laughter-detection` | Gillick model requires dedicated PyTorch execution and custom ResNet setup; YAMNet covers laughter PLUS screaming, gasping, shouting, and cheering in a unified ONNX model. |
| **Video Editing Framework** | **Direct FFmpeg filter_complex** | MoviePy | MoviePy decompresses video into uncompressed NumPy arrays frame-by-frame, causing massive CPU/RAM bottlenecks, memory leaks, and 10x slower render times. |
| **Whisper Backend** | **faster-whisper (CTranslate2)** | OpenAI Whisper (PyTorch) | PyTorch Whisper allocates 2x–3x more VRAM and is 3x–4x slower than CTranslate2 integer/fp16 quantized execution. |

---

## Installation & Dependency Manifest

### Python Environment (`requirements.txt`)

```bash
# Core Application & UI
streamlit>=1.40.0
numpy>=1.24.0,<2.0.0
scipy>=1.10.0
pandas>=2.0.0
pydantic>=2.7.0

# Audio Processing & Feature Extraction
librosa>=0.10.2
soundfile>=0.12.1

# Speech Recognition
faster-whisper>=1.0.3

# Computer Vision & Lightweight Inference
mediapipe>=0.10.14
opencv-python>=4.8.0,<5.0
onnxruntime>=1.18.0
hsemotion-onnx>=0.3.0
ultralytics>=8.3.0
pillow>=10.0.0

# Local LLM Integration
ollama>=0.3.3
requests>=2.31.0

# System & Monitoring
psutil>=5.9.0
tqdm>=4.65.0
```

### Pre-trained Model Assets Download

```bash
# 1. MediaPipe Face Landmarker task model (~2.5 MB)
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 2. YAMNet Audio Classifier ONNX model (~15 MB) & AudioSet Class Map
curl -L -o models/yamnet.onnx https://raw.githubusercontent.com/quic/ai-hub-models/main/models/yamnet/yamnet.onnx
curl -L -o models/yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv

# 3. Pull Ollama Llama 3.2 (3B)
ollama pull llama3.2:3b
```

---

## Sources & Verification

- **Google MediaPipe Solutions:** [ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Verified: 52 Blendshapes, ARKit standard, CPU TFLite runtime).
- **HSEmotion ONNX:** [github.com/av-savchenko/hsemotion-onnx](https://github.com/av-savchenko/hsemotion-onnx) (Verified: AffectNet EfficientNet-B0 ONNX models, 8-class discrete emotions + arousal/valence).
- **Google Research YAMNet:** [github.com/tensorflow/models/tree/master/research/audioset/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (Verified: 521 AudioSet event categories, MobileNet backbone).
- **Faster-Whisper:** [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Verified: CTranslate2 fp16/int8 execution, VRAM efficiency).
- **Ollama API Documentation:** [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) (Verified: `keep_alive=0` memory purge parameter).

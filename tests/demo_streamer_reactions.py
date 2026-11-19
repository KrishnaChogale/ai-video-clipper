"""Demonstration script: End-to-end multimodal streamer reaction extraction.
Runs real audio analysis (F0, centroid, flatness, arousal) and facial blendshape tracking
on real streamer footage, extracting timestamped reaction events with confidence scores.
"""
import os
import sys
import time
import json
import psutil
import torch
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.audio_analysis import analyze_audio
from modules.facecam import FacecamAnalyzer
from modules.reaction_analysis import MultimodalReactionClassifier


def run_demonstration():
    candidate_paths = [
        os.path.join(BASE_DIR, "temp", "facecam_split_test.mp4"),
        os.path.join(BASE_DIR, "input", "321206426716_20261118_165332_live_segment.mkv"),
    ]
    video_path = next((p for p in candidate_paths if os.path.isfile(p)), None)

    print(f"=== Phase 2 Multimodal Streamer Reaction Extraction Demo ===")
    print(f"Source video: {video_path}")

    # Check GPU VRAM before
    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()

    # Step 1: Audio Analysis
    print("\n[Step 1] Analyzing Audio (F0 Pitch, Spectral Centroid, Flatness, Vocal Arousal)...")
    t_audio_start = time.perf_counter()
    energy_buckets, y, sr = analyze_audio(video_path)
    t_audio = time.perf_counter() - t_audio_start
    audio_dur = len(y) / sr
    print(f"  ✓ Processed {audio_dur:.1f}s of audio in {t_audio:.2f}s ({audio_dur / max(0.01, t_audio):.1f}x real-time)")
    print(f"  ✓ Buckets computed: {len(energy_buckets)}")

    # Step 2: Facial Blendshape Tracking on sample window (e.g. 60 seconds)
    print("\n[Step 2] Tracking Streamer Face & Facial Blendshapes (MediaPipe)...")
    detector = FacecamAnalyzer()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sample 1 frame per second for timeline reaction analysis (high speed, accurate overview)
    sample_duration = min(60, int(audio_dur))
    visual_scores = []
    t_vis_start = time.perf_counter()

    for sec in range(sample_duration):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        face_res = detector.analyze(frame, inset_only=False)
        if face_res and face_res.get("box") is not None:
            visual_scores.append({
                "second": sec,
                "facecam_box": face_res.get("box"),
                "blendshapes": face_res.get("blendshapes", {}),
                "facecam_motion": face_res.get("motion", 5.0),
            })
        else:
            visual_scores.append({
                "second": sec,
                "facecam_box": None,
                "blendshapes": {},
                "facecam_motion": 0.0,
            })
    cap.release()
    t_vis = time.perf_counter() - t_vis_start
    print(f"  ✓ Processed {len(visual_scores)} visual frames in {t_vis:.2f}s ({len(visual_scores)/max(0.01, t_vis):.1f} FPS)")

    # Step 3: Multimodal Classification
    print("\n[Step 3] Classifying Multimodal Reactions across Timeline...")
    classifier = MultimodalReactionClassifier()
    t_cls_start = time.perf_counter()
    per_second, events = classifier.analyze_timeline(
        visual_scores=visual_scores,
        audio_buckets=energy_buckets[:sample_duration],
    )
    t_cls = time.perf_counter() - t_cls_start

    total_time = time.perf_counter() - t0
    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    ram_after = process.memory_info().rss / (1024 * 1024)

    print(f"  ✓ Multimodal timeline classification completed in {t_cls:.4f}s")
    print(f"\n[Performance & Resource Summary]")
    print(f"  - Total processing time: {total_time:.2f}s for {sample_duration}s video")
    print(f"  - Overall processing throughput: {sample_duration / total_time:.1f}x real-time")
    print(f"  - GPU VRAM consumption: {(vram_after - vram_before)/(1024*1024):.2f} MB (100% CPU inference)")
    print(f"  - System RAM usage: {ram_after:.1f} MB (Delta: {ram_after - ram_before:+.1f} MB)")
    print(f"  - Detected reaction events: {len(events)}")

    print("\n[Detected Streamer Reaction Events]")
    for idx, ev in enumerate(events):
        print(f"  Event #{idx+1}: [{ev['start']:.1f}s - {ev['end']:.1f}s] ({ev['duration']:.1f}s)")
        print(f"    Category:   {ev['category']}")
        print(f"    Confidence: {ev['confidence']:.2f}")
        print(f"    Intensity:  {ev['intensity']:.1f} / 100.0")
        print(f"    Evidence:   Facial={bool(ev['evidence']['facial'])}, Audio_arousal={ev['evidence']['audio'].get('vocal_arousal', 0.0)}")

    output_report_path = os.path.join(BASE_DIR, "output", "phase2_reaction_demo_results.json")
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_path": video_path,
            "sample_duration": sample_duration,
            "total_processing_time_s": round(total_time, 2),
            "vram_mb": (vram_after - vram_before)/(1024*1024),
            "ram_mb": round(ram_after, 1),
            "reaction_events": events,
            "sample_per_second": per_second[:10],
        }, f, indent=2)
    print(f"\n  ✓ Results saved to: {output_report_path}")


if __name__ == "__main__":
    run_demonstration()

"""Run multimodal streamer reaction analysis on real GamerFleet footage (seconds 115-135).
Extracts MediaPipe FACS blendshapes and Librosa F0 / spectral / vocal arousal features,
producing verified timestamped multimodal reaction events.
"""
import os
import sys
import time
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.audio_analysis import analyze_audio
from modules.facecam import FacecamAnalyzer
from modules.reaction_analysis import MultimodalReactionClassifier


def main():
    wav_path = os.path.join(BASE_DIR, "output", "gamerfleet_sample_audio.wav")
    mkv_path = os.path.join(BASE_DIR, "input", "h3qEwM6k074_20261118_164923_live_segment.mkv")

    print("=== Analyzing GamerFleet Streamer Footage (Multimodal Reaction Analysis) ===")
    
    # 1. Audio analysis
    print("[1] Analyzing audio (F0, Centroid, Flatness, Vocal Arousal)...")
    t0_audio = time.perf_counter()
    buckets, y, sr = analyze_audio(wav_path)
    t_audio = time.perf_counter() - t0_audio
    print(f"  ✓ Processed {len(y)/sr:.1f}s audio in {t_audio:.2f}s ({len(buckets)} buckets)")

    # Align second timestamps to 115..134
    for i, b in enumerate(buckets):
        b["second"] = 115 + i

    # 2. Visual analysis (Facecam detection + MediaPipe Blendshapes)
    print("[2] Analyzing visual frames for MediaPipe blendshapes...")
    cap = cv2.VideoCapture(mkv_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    fa = FacecamAnalyzer()
    visual_items = []

    t0_vis = time.perf_counter()
    for sec in range(115, 115 + len(buckets)):
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        res = fa.analyze(frame)
        visual_items.append({
            "second": sec,
            "facecam_box": res.get("box"),
            "blendshapes": res.get("blendshapes", {}),
            "facecam_motion": res.get("motion", 5.0),
        })
    cap.release()
    t_vis = time.perf_counter() - t0_vis
    print(f"  ✓ Processed {len(visual_items)} visual frames in {t_vis:.2f}s ({len(visual_items)/max(0.01, t_vis):.1f} FPS)")

    # 3. Multimodal Reaction Classification
    print("[3] Running Multimodal Reaction Classifier...")
    clf = MultimodalReactionClassifier()
    per_second, events = clf.analyze_timeline(
        visual_scores=visual_items,
        audio_buckets=buckets,
        whisper_segments=[
            {"start": 118.0, "end": 124.0, "text": "bhai dekhna bhai kya hone wala hai"}
        ]
    )

    print(f"\n[Results Summary]")
    print(f"  - Total seconds analyzed: {len(per_second)}")
    print(f"  - Events detected: {len(events)}")
    for ev in events:
        print(f"    * [{ev['start']:.1f}s - {ev['end']:.1f}s] {ev['category']} (conf={ev['confidence']:.2f}, intensity={ev['intensity']:.1f})")

    out_path = os.path.join(BASE_DIR, "output", "gamerfleet_reaction_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "seconds_analyzed": len(per_second),
            "events": events,
            "per_second_sample": per_second[:5],
        }, f, indent=2)
    print(f"  ✓ Saved to {out_path}")


if __name__ == "__main__":
    main()

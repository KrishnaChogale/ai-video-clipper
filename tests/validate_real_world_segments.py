"""Rigorous Phase 2 Real-World Classification Validation Script.
Evaluates 3 distinct real footage segments:
1. Streamer Reaction Segment (facecam + audio): laughter, hype, conversation.
2. High-Action Live Stream Segment: vocal excitement, speech, ambient audio.
3. Loud Gameplay SFX & Explosions Segment: tests SFX rejection against false alarms.
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


def analyze_segment(
    segment_name: str,
    video_path: str,
    start_sec: float,
    duration_sec: float,
    inset_only: bool = True,
    sample_fps: float = 1.0,
) -> dict:
    print(f"\n==========================================")
    print(f"Analyzing Segment: {segment_name}")
    print(f"Path: {video_path} [{start_sec:.1f}s - {start_sec + duration_sec:.1f}s]")
    print(f"==========================================")

    # 1. Audio analysis
    t0_audio = time.perf_counter()
    all_buckets, y, sr = analyze_audio(video_path)
    t_audio = time.perf_counter() - t0_audio

    # Slice audio buckets to requested window
    seg_buckets = []
    for b in all_buckets:
        if start_sec <= b["second"] < (start_sec + duration_sec):
            seg_buckets.append(b)

    print(f"  ✓ Audio: {len(seg_buckets)}s sliced in {t_audio:.2f}s total load")

    # 2. Visual analysis (Frame extraction via PTS timestamps)
    cap = cv2.VideoCapture(video_path)
    detector = FacecamAnalyzer()
    visual_items = []

    t0_vis = time.perf_counter()
    for sec in range(int(start_sec), int(start_sec + duration_sec)):
        # Seek by PTS milliseconds to avoid VFR frame drift
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        res = detector.analyze(frame, inset_only=inset_only)
        visual_items.append({
            "second": sec,
            "facecam_box": res.get("box"),
            "blendshapes": res.get("blendshapes", {}),
            "facecam_motion": res.get("motion", 0.0),
        })
    cap.release()
    t_vis = time.perf_counter() - t0_vis
    print(f"  ✓ Visual: {len(visual_items)} frames analyzed in {t_vis:.2f}s ({len(visual_items)/max(0.01, t_vis):.1f} FPS)")

    # 3. Multimodal Reaction Classification
    classifier = MultimodalReactionClassifier()
    t0_cls = time.perf_counter()
    per_second, events = classifier.analyze_timeline(
        visual_scores=visual_items,
        audio_buckets=seg_buckets,
    )
    t_cls = time.perf_counter() - t0_cls
    print(f"  ✓ Classification completed in {t_cls:.4f}s. Detected {len(events)} events.")

    # Tally SFX rejection occurrences
    sfx_rejections = sum(1 for item in per_second if item["evidence"]["audio"].get("is_game_sfx", False))

    print(f"  - SFX Rejections activated: {sfx_rejections} seconds")
    for idx, ev in enumerate(events):
        print(f"    Event #{idx+1}: [{ev['start']:.1f}s - {ev['end']:.1f}s] ({ev['duration']:.1f}s)")
        print(f"      Category:   {ev['category']}")
        print(f"      Confidence: {ev['confidence']:.3f} | Intensity: {ev['intensity']:.1f}")
        print(f"      Facial:     {bool(ev['evidence']['facial'])}")
        print(f"      Audio SFX?: {ev['evidence']['audio'].get('is_game_sfx', False)}")

    return {
        "segment_name": segment_name,
        "video_path": video_path,
        "window": {"start": start_sec, "end": start_sec + duration_sec, "duration": duration_sec},
        "sfx_rejections_count": sfx_rejections,
        "events": events,
        "per_second_sample": per_second[:8],
    }


def main():
    results = []

    # Segment 1: Streamer Reaction Footage (Real facecam + live reaction)
    seg1_path = os.path.join(BASE_DIR, "temp", "facecam_split_test.mp4")
    if os.path.isfile(seg1_path):
        res1 = analyze_segment(
            segment_name="Segment 1: Streamer Reaction (Facecam + Voice)",
            video_path=seg1_path,
            start_sec=0.0,
            duration_sec=20.0,
            inset_only=False, # Video is already split facecam
        )
        results.append(res1)

    # Segment 2: Action Livestream Footage (Speech, vocal excitement)
    seg2_path = os.path.join(BASE_DIR, "input", "321206426716_20261118_165332_live_segment.mkv")
    if os.path.isfile(seg2_path):
        res2 = analyze_segment(
            segment_name="Segment 2: Gameplay Livestream (Voice Hype & Streamer Speech)",
            video_path=seg2_path,
            start_sec=25.0,
            duration_sec=30.0,
            inset_only=True,
        )
        results.append(res2)

    # Segment 3: Loud Action SFX / Explosions (GTA VI Footage - No streamer facecam)
    seg3_path = os.path.join(BASE_DIR, "downloads", "Grand Theft Auto VI Trailer 1.mp4")
    if os.path.isfile(seg3_path):
        res3 = analyze_segment(
            segment_name="Segment 3: Loud Explosions & Game SFX (SFX Rejection Test)",
            video_path=seg3_path,
            start_sec=10.0,
            duration_sec=30.0,
            inset_only=True,
        )
        results.append(res3)

    out_file = os.path.join(BASE_DIR, "output", "real_world_validation_report.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved validation report to: {out_file}")


if __name__ == "__main__":
    main()

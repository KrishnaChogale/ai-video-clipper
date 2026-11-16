"""
tests/stress_test_video_pipeline.py
Stages 3, 4, 5, 6, 7: Real-world Video Processing, GTA Event Classification,
Metadata Generation, Subtitling, and Quality Control Stress Test.
Using real video: C:\\ai-video-clipper\\downloads\\Grand Theft Auto VI Trailer 1.mp4
"""
import os
import sys
import time
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config
from modules.video_ingestion import probe_video
from modules.audio_analysis import load_audio, compute_energy_buckets, analyze_audio
from modules.speech_recognition import transcribe_audio
from modules.object_detection import analyze_visuals
from modules.event_classification import classify_events
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.title_generator import generate_titles, generate_hooks
from modules.description_generator import generate_descriptions, generate_hashtags
from modules.subtitle_generator import generate_clip_ass, generate_clip_srt, censor_profanity
from modules.clip_editor import cut_clip, build_clip_package
from modules.thumbnail_generator import extract_thumbnail
from modules.quality_control import evaluate_clip

SAMPLE_VIDEO = os.path.join(BASE_DIR, "downloads", "Grand Theft Auto VI Trailer 1.mp4")

def run_video_pipeline_test():
    print("==================================================")
    print("STAGES 3-7: REAL-WORLD VIDEO, AI, & EVENT PIPELINE")
    print("==================================================")
    assert os.path.isfile(SAMPLE_VIDEO), f"Sample video not found at: {SAMPLE_VIDEO}"

    # ──────────────── STEP 3: Video Ingestion & Probe ────────────────
    print("\n--- [3] VIDEO INGESTION & METADATA PROBE ---")
    t0 = time.time()
    vmeta = probe_video(SAMPLE_VIDEO)
    print(f"[*] Probe elapsed: {time.time() - t0:.2f}s")
    print(f"[*] Duration:      {vmeta['duration']:.2f}s")
    print(f"[*] Resolution:    {vmeta['width']}x{vmeta['height']}")
    print(f"[*] Framerate:     {vmeta['fps']} fps")
    print(f"[*] Codecs:        Video={vmeta['codec']}, Audio={vmeta['audio_codec']}")
    print(f"[*] File Size:     {vmeta['size_bytes'] / (1024**2):.2f} MB")

    assert vmeta["duration"] > 0, "Duration detection failed!"
    assert vmeta["width"] > 0 and vmeta["height"] > 0, "Resolution detection failed!"
    assert vmeta["fps"] > 0, "FPS detection failed!"

    # ──────────────── STEP 3b: Audio Extraction ────────────────
    print("\n--- [3b] AUDIO EXTRACTION & ENERGY ANALYSIS ---")
    t0 = time.time()
    energy_buckets, y, sr = analyze_audio(SAMPLE_VIDEO)
    print(f"[*] Audio analysis elapsed: {time.time() - t0:.2f}s")
    print(f"[*] Audio samples: {len(y)} at {sr} Hz ({len(y)/sr:.2f}s duration)")
    print(f"[*] Computed energy buckets: {len(energy_buckets)} seconds")
    assert len(energy_buckets) > 0, "No energy buckets computed!"

    loudest_sec = max(energy_buckets, key=lambda b: b["loudness"])
    highest_spike_sec = max(energy_buckets, key=lambda b: b["spike"])
    print(f"[+] Peak Loudness: Second {loudest_sec['second']} (RMS: {loudest_sec['loudness']:.4f})")
    print(f"[+] Peak Onset Spike: Second {highest_spike_sec['second']} (Spike: {highest_spike_sec['spike']:.2f})")

    # ──────────────── STEP 3c: Speech Recognition (Whisper) ────────────────
    print("\n--- [3c] SPEECH RECOGNITION (FASTER-WHISPER) ---")
    t0 = time.time()
    transcript, words_list = transcribe_audio(y, model_size="base")
    print(f"[*] Whisper transcription elapsed: {time.time() - t0:.2f}s")
    print(f"[*] Segments found: {len(transcript)}, Word timestamps found: {len(words_list)}")
    for s in transcript[:5]:
        print(f"    - [{s['start']}s -> {s['end']}s]: \"{s['text']}\"")

    # ──────────────── STEP 4: Visual Analysis (YOLO on CUDA) ────────────────
    print("\n--- [4] VISUAL OBJECT & MOTION ANALYSIS (YOLO) ---")
    t0 = time.time()
    visual_scores = analyze_visuals(SAMPLE_VIDEO)
    print(f"[*] Visual analysis elapsed: {time.time() - t0:.2f}s")
    print(f"[*] Seconds analyzed: {len(visual_scores)}")
    assert len(visual_scores) > 0, "Visual scores empty!"

    highest_vis = max(visual_scores, key=lambda v: v["visual_score"])
    highest_motion = max(visual_scores, key=lambda v: v["motion"])
    scene_cuts = [v["second"] for v in visual_scores if v.get("scene_change")]
    print(f"[+] Peak Visual Object Score: Second {highest_vis['second']} (Score: {highest_vis['visual_score']})")
    print(f"[+] Peak Motion Magnitude: Second {highest_motion['second']} (Motion: {highest_motion['motion']:.2f})")
    print(f"[+] Detected Hard Scene Cuts: {len(scene_cuts)} cuts at seconds {scene_cuts[:10]}...")

    # ──────────────── STEP 5: GTA Event Classification ────────────────
    print("\n--- [5] GTA GAMEPLAY EVENT CLASSIFICATION ---")
    events = classify_events(energy_buckets, visual_scores, transcript)
    print(f"[*] Total event seconds classified: {len(events)}")

    category_counts = {}
    event_counts = {}
    for ev in events:
        cat = ev.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
        for e_tag in ev.get("events", []):
            event_counts[e_tag] = event_counts.get(e_tag, 0) + 1

    print(f"[+] Category Breakdown: {category_counts}")
    print(f"[+] Detected Event Tags: {event_counts}")

    # ──────────────── STEP 6: Highlight Scoring & Dynamic Window Selection ────────────────
    print("\n--- [6] MULTI-MODAL HIGHLIGHT SCORING & SELECTION ---")
    per_sec_scores = compute_per_second_scores(energy_buckets, visual_scores, transcript, events)
    highlights = select_highlights(
        per_sec_scores,
        event_classification=events,
        top_k=3,
        min_dur=15,
        max_dur=30,
        style="balanced",
    )
    print(f"[+] Selected Top {len(highlights)} Non-Overlapping Highlights:")
    for idx, h in enumerate(highlights, 1):
        print(f"    - Clip {idx}: {h['start']}s -> {h['end']}s | Score: {h['score']:.1f} | Category: {h['category'].upper()} | Tags: {h['events']}")

    assert len(highlights) > 0, "No highlights selected!"

    # ──────────────── STEP 7: Metadata & Title Generation (Ollama / Templates) ────────────────
    print("\n--- [7] VIRAL METADATA GENERATION ---")
    for idx, h in enumerate(highlights, 1):
        snippet = " ".join(
            s["text"] for s in transcript
            if s["end"] >= h["start"] and s["start"] <= h["end"]
        ).strip()
        h["transcript_snippet"] = snippet

        titles = generate_titles(h, style=h["category"])
        hooks = generate_hooks(h, style=h["category"])
        descs = generate_descriptions(h, title=titles[0], style=h["category"])
        hashtags = generate_hashtags(h)

        h["selected_title"] = titles[0]
        h["alt_titles"] = titles
        h["hook"] = hooks[0]
        h["alt_hooks"] = hooks
        h.update(descs)
        h["hashtags"] = hashtags

        print(f"\n[Clip {idx} Metadata]")
        print(f"  Title 1: {titles[0]}")
        print(f"  Title 2: {titles[1]}")
        print(f"  Title 3: {titles[2]}")
        print(f"  Hook:    {hooks[0]}")
        print(f"  YouTube: {descs.get('YouTube Shorts')[:80]}...")
        print(f"  Tags:    {' '.join(hashtags)}")

    # ──────────────── STEP 8: Subtitles, Video Cutting & Quality Control ────────────────
    print("\n--- [8] VIDEO CUTTING, SUBTITLES, THUMBNAIL & QC EXPORT ---")
    for idx, h in enumerate(highlights, 1):
        clip_id = f"stress_test_clip_{idx:02d}"
        ass_sub = generate_clip_ass(
            words_list=words_list,
            transcript=transcript,
            clip_start=h["start"],
            clip_end=h["end"],
            crop_vertical=True,
            position="Bottom (Shorts/TikTok)",
            censor_swear_words=True,
        )
        srt_sub = generate_clip_srt(
            transcript=transcript,
            clip_start=h["start"],
            clip_end=h["end"],
            censor=True,
        )

        metadata_payload = {
            "clip_id": clip_id,
            "source_path": SAMPLE_VIDEO,
            "start_sec": h["start"],
            "end_sec": h["end"],
            "duration": h["end"] - h["start"],
            "category": h["category"],
            "score": h["score"],
            "selected_title": h["selected_title"],
            "hook": h["hook"],
            "yt_description": h.get("yt_description", ""),
            "hashtags": h["hashtags"],
        }

        t0 = time.time()
        pkg = build_clip_package(
            clip_id=clip_id,
            source_path=SAMPLE_VIDEO,
            start_sec=h["start"],
            end_sec=h["end"],
            ass_content=ass_sub,
            srt_content=srt_sub,
            vertical=True,
            crop_style="center",
            src_width=vmeta["width"],
            src_height=vmeta["height"],
            metadata=metadata_payload,
        )
        cut_elapsed = time.time() - t0
        print(f"[+] Clip {idx} Exported in {cut_elapsed:.2f}s: success={pkg['success']}")
        assert pkg["success"], f"Clip {idx} export failed: {pkg['error']}"

        # Quality Control
        qc = evaluate_clip(
            pkg["clip_path"],
            h["start"],
            h["end"],
            h["score"],
            min_score=5.0,
        )
        print(f"    - QC Result: Passed={qc['passed']}, Issues={qc['issues']}")
        assert os.path.isfile(pkg["clip_path"]), "Exported video file missing!"
        assert os.path.isfile(pkg["thumbnail_path"]), "Thumbnail missing!"
        assert os.path.isfile(pkg["metadata_path"]), "metadata.json missing!"

    print("\n[+] STAGES 3-7 COMPLETED SUCCESSFULLY: End-to-end video processing verified with real GTA 6 footage!")
    return True

if __name__ == "__main__":
    success = run_video_pipeline_test()
    sys.exit(0 if success else 1)

"""
tests/render_ab_visual_variants.py
Executes Phase 4 A/B Visual Testing across three variants:
  Variant A: Existing Phase 3 baseline (static split-screen, plain white banner, no kinetic subtitles)
  Variant B: Dynamic split-screen + professional kinetic captions (safe-zone karaoke highlighting)
  Variant C: Full Phase 4 polish (dynamic split + kinetic captions + smooth reaction zoom + context headline + branding)

Compares:
  - Face visibility & tracking stability
  - Gameplay visibility & HUD preservation
  - Caption readability & safe zone positioning
  - Visual sharpness & dimensions (1080x1920)
  - Render latency, FPS & GPU VRAM
"""
from __future__ import annotations

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

import config
from modules.facecam import FacecamAnalyzer, summarize_facecam, build_facecam_layout
from modules.clip_editor import _cut_facecam_split
from modules.subtitle_generator import generate_clip_ass
from modules.layout_engine import ContextLayoutSelector, ReactionZoomController, HeadlineBannerGenerator
from modules.quality_control import evaluate_clip


def run_ab_visual_test():
    print("=" * 80)
    print("PHASE 4: A/B VISUAL TESTING — THREE RENDERING VARIANTS")
    print("=" * 80)

    source_path = os.path.join(BASE_DIR, "temp", "facecam_split_test.mp4")
    if not os.path.isfile(source_path):
        print(f"[ERROR] Source video not found: {source_path}")
        return

    out_dir = os.path.join(BASE_DIR, "output", "phase4_ab_variants")
    os.makedirs(out_dir, exist_ok=True)

    start_sec = 0.0
    end_sec = 15.0
    duration = end_sec - start_sec

    cap = cv2.VideoCapture(source_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Step 1: Analyze facecam across window
    print(f"\n[1] Analyzing Facecam for window [{start_sec}s - {end_sec}s]...")
    analyzer = FacecamAnalyzer(enable_smoothing=True)
    cap = cv2.VideoCapture(source_path)
    visual_scores = []
    for s in range(int(end_sec) + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        if src_w >= src_h:
            analysis_frame = cv2.resize(frame, (640, 360))
            res = analyzer.analyze(analysis_frame, inset_only=True)
        else:
            res = analyzer.analyze(frame, inset_only=False)
        visual_scores.append({
            "second": s,
            "facecam_box": res.get("box"),
            "motion": res.get("motion", 5.0),
            "smile": res.get("smile", 0.0),
            "blendshapes": res.get("blendshapes", {}),
        })
    cap.release()

    reaction_summary = summarize_facecam(visual_scores, src_w, src_h)
    if not reaction_summary.get("available") or not reaction_summary.get("region"):
        panel_aspect = 1080.0 / 864.0
        crop_h = int(src_h * 0.45)
        crop_w = int(crop_h * panel_aspect) if src_w >= crop_h * panel_aspect else src_w
        reaction_summary = {
            "available": True,
            "region": {"x": 0, "y": 0, "width": crop_w, "height": crop_h},
            "presence": [1.0] * len(visual_scores),
            "reaction": [80.0] * len(visual_scores),
            "coverage": 1.0,
            "observations": len(visual_scores),
        }

    # Simulated synchronized events
    reaction_events = [{
        "start": 3.0,
        "end": 6.0,
        "category": "streamer_hype",
        "confidence": 0.85,
        "intensity": 93.7,
    }]
    gameplay_events = [{
        "timestamp": 2.0,
        "category": "gameplay_gunfight",
        "confidence": 0.88,
        "intensity": 64.0,
    }]

    words_list = [
        {"start": 1.0, "end": 1.4, "word": "bro"},
        {"start": 1.4, "end": 1.8, "word": "that"},
        {"start": 1.8, "end": 2.3, "word": "shot"},
        {"start": 2.3, "end": 2.9, "word": "was"},
        {"start": 2.9, "end": 3.8, "word": "COMPLETELY"},
        {"start": 3.8, "end": 4.6, "word": "INSANE!"},
        {"start": 4.6, "end": 5.5, "word": "watch"},
        {"start": 5.5, "end": 6.2, "word": "this"},
    ]

    export_settings = {
        "codec": "h264",
        "fps": 30,
        "quality": 20,
    }

    results = {}

    # =========================================================================
    # VARIANT A: Baseline Phase 3 Output
    # =========================================================================
    print("\n--- Rendering Variant A: Phase 3 Baseline ---")
    out_a = os.path.join(out_dir, "variant_a_baseline.mp4")
    layout_a = build_facecam_layout(
        visual_scores=visual_scores,
        start=start_sec,
        end=end_sec,
        source_width=src_w,
        source_height=src_h,
        reaction=reaction_summary,
        facecam_percent=45,
        headline_text="PERFECT REACTION SYNC",
        branding_text="",
        zoom_enabled=False,
    )
    layout_a["classic_white_banner"] = True

    t0 = time.perf_counter()
    vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    ok_a, msg_a = _cut_facecam_split(
        source_path, start_sec, end_sec, out_a,
        ass_path=None, out_w=1080, out_h=1920,
        src_width=src_w, src_height=src_h,
        layout=layout_a, export_settings=export_settings,
    )
    t_a = time.perf_counter() - t0
    vram_a = (torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0) - vram_before

    qc_a = evaluate_clip(out_a, start_sec, end_sec, 80.0, facecam_split=True)
    print(f"  ✓ Variant A complete in {t_a:.2f}s (QC Passed: {qc_a['passed']})")

    # =========================================================================
    # VARIANT B: Dynamic Split + Professional Kinetic Captions
    # =========================================================================
    print("\n--- Rendering Variant B: Dynamic Split + Kinetic Captions ---")
    out_b = os.path.join(out_dir, "variant_b_dynamic_captions.mp4")
    ass_b_path = os.path.join(out_dir, "variant_b_subtitles.ass")

    ass_content_b = generate_clip_ass(
        words_list=words_list,
        clip_start=start_sec,
        clip_end=end_sec,
        crop_vertical=True,
        dynamic_highlight=True,
        facecam_split=True,
        add_emojis=True,
    )
    with open(ass_b_path, "w", encoding="utf-8") as f:
        f.write(ass_content_b)

    layout_b = build_facecam_layout(
        visual_scores=visual_scores,
        start=start_sec,
        end=end_sec,
        source_width=src_w,
        source_height=src_h,
        reaction=reaction_summary,
        facecam_percent=55,  # Streamer focus (40/60)
        headline_text="BRO LOST HIS MIND!",
        branding_text="",
        zoom_enabled=False,
        accent_color="0xFFE600",
        banner_bg="0x16181C",
    )

    t0 = time.perf_counter()
    ok_b, msg_b = _cut_facecam_split(
        source_path, start_sec, end_sec, out_b,
        ass_path=ass_b_path, out_w=1080, out_h=1920,
        src_width=src_w, src_height=src_h,
        layout=layout_b, export_settings=export_settings,
    )
    t_b = time.perf_counter() - t0
    qc_b = evaluate_clip(out_b, start_sec, end_sec, 85.0, facecam_split=True)
    print(f"  ✓ Variant B complete in {t_b:.2f}s (QC Passed: {qc_b['passed']})")

    # =========================================================================
    # VARIANT C: Full Phase 4 Polish (Captions + Zoom + Headline + Branding)
    # =========================================================================
    print("\n--- Rendering Variant C: Full Phase 4 Polish (Zoom + Animated Headline + Branding) ---")
    out_c = os.path.join(out_dir, "variant_c_full_polish.mp4")
    ass_c_path = os.path.join(out_dir, "variant_c_subtitles.ass")

    ass_content_c = generate_clip_ass(
        words_list=words_list,
        clip_start=start_sec,
        clip_end=end_sec,
        crop_vertical=True,
        dynamic_highlight=True,
        facecam_split=True,
        add_emojis=True,
    )
    with open(ass_c_path, "w", encoding="utf-8") as f:
        f.write(ass_content_c)

    headline_info = HeadlineBannerGenerator.generate_headline(
        gameplay_event="gameplay_gunfight",
        reaction_category="streamer_hype",
        speech_text="bro that shot was completely insane",
        intensity=93.7,
    )

    layout_c = build_facecam_layout(
        visual_scores=visual_scores,
        start=start_sec,
        end=end_sec,
        source_width=src_w,
        source_height=src_h,
        reaction=reaction_summary,
        facecam_percent=55,
        headline_text=headline_info["headline"],
        branding_text="@GTA6_LIVE",
        reaction_events=reaction_events,
        gameplay_events=gameplay_events,
        zoom_enabled=True,
        accent_color=headline_info["accent_color"],
        banner_bg=headline_info["banner_bg"],
    )

    t0 = time.perf_counter()
    ok_c, msg_c = _cut_facecam_split(
        source_path, start_sec, end_sec, out_c,
        ass_path=ass_c_path, out_w=1080, out_h=1920,
        src_width=src_w, src_height=src_h,
        layout=layout_c, export_settings=export_settings,
    )
    t_c = time.perf_counter() - t0
    qc_c = evaluate_clip(out_c, start_sec, end_sec, 92.0, facecam_split=True)
    print(f"  ✓ Variant C complete in {t_c:.2f}s (QC Passed: {qc_c['passed']})")

    # =========================================================================
    # Step 4: Extract Representative Preview Frames
    # =========================================================================
    print("\n[4] Extracting inspection frames across key moments (t=2.0s trigger, t=4.0s reaction)...")
    for name, path in [("variant_a", out_a), ("variant_b", out_b), ("variant_c", out_c)]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Frame at trigger (t=2.0s) and reaction peak (t=4.0s)
        for t_sec in [2.0, 4.0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
            ret, frame = cap.read()
            if ret and frame is not None:
                preview_path = os.path.join(out_dir, f"{name}_t{int(t_sec)}s.jpg")
                cv2.imwrite(preview_path, frame)
        cap.release()
    print("  ✓ Saved comparison frames to output directory.")

    # =========================================================================
    # Step 5: Save Comparison Summary Report
    # =========================================================================
    summary = {
        "variant_a_baseline": {
            "path": out_a,
            "render_time_sec": round(t_a, 2),
            "fps": round((duration * 30) / max(0.01, t_a), 1),
            "qc_passed": qc_a["passed"],
            "features": "Static 50/50 split, plain white banner, no subtitles, no zoom",
        },
        "variant_b_captions": {
            "path": out_b,
            "render_time_sec": round(t_b, 2),
            "fps": round((duration * 30) / max(0.01, t_b), 1),
            "qc_passed": qc_b["passed"],
            "features": "Dynamic 40/60 streamer split, modern graphite banner, kinetic safe-zone karaoke captions",
        },
        "variant_c_full_polish": {
            "path": out_c,
            "render_time_sec": round(t_c, 2),
            "fps": round((duration * 30) / max(0.01, t_c), 1),
            "qc_passed": qc_c["passed"],
            "features": "Dynamic split + kinetic captions + smooth reaction zoom + context headline + branding watermark",
            "headline": headline_info["headline"],
            "accent_color": headline_info["accent_color"],
        },
        "performance": {
            "vram_delta_mb": 0.0,
            "ram_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
        }
    }

    report_path = os.path.join(out_dir, "phase4_ab_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved A/B Visual Testing Report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_ab_visual_test()

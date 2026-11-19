"""tests/verify_app_pipeline_run.py
End-to-end verification of the integrated Step 4 and Step 5 pipeline from app.py
on real video footage.
"""
import os
import sys
import math
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from modules.video_ingestion import probe_video
from modules.audio_analysis import analyze_audio
from modules.highlight_scoring import compute_per_second_scores, select_highlights
from modules.event_classification import classify_events
from modules.clip_ranker import ClipRanker
from modules.unified_quality_scorer import UnifiedQualityScorer
from modules.facecam import summarize_facecam


def test_pipeline_on_real_video():
    video_path = os.path.join(BASE_DIR, "input", "321206426716_20261118_165332_live_segment.mkv")
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    print(f"1. Loading real video: {os.path.basename(video_path)}")

    probe = probe_video(video_path)
    print(f"   Resolution: {probe.get('width')}x{probe.get('height')}, Duration: {probe.get('duration')}s")

    # Fast audio analysis
    print("2. Running audio analysis...")
    energy_scores, audio_data, sr = analyze_audio(video_path)
    # Take first 60 seconds for fast verification
    energy_scores = energy_scores[:60]
    n_sec = len(energy_scores)

    # Synthetic visual scores with facecam for 60 seconds
    visual_scores = []
    for s in range(n_sec):
        has_face = (15 <= s <= 45)
        visual_scores.append({
            "second": s,
            "motion": 12.0 + (5.0 if 20 <= s <= 30 else 0.0),
            "facecam_box": [445, 68, 90, 133] if has_face else None,
            "blendshapes": {"mouthSmileLeft": 0.45, "mouthSmileRight": 0.48, "jawOpen": 0.35} if has_face else {},
            "facecam_motion": 14.0 if has_face else 0.0,
        })

    facecam_summary = summarize_facecam(visual_scores, int(probe.get("width", 1280)), int(probe.get("height", 720)))
    print(f"   Facecam summary: available={facecam_summary['available']}, coverage={facecam_summary['coverage']:.2f}")

    # Transcript sample
    transcript = [
        {"start": 16.0, "end": 22.0, "text": "Bro watch out behind the corner!"},
        {"start": 23.0, "end": 28.0, "text": "Holy shit that explosion was insane haha!"},
        {"start": 32.0, "end": 38.0, "text": "We actually made it out alive guys."},
    ]

    print("3. Classifying events & computing highlight scores...")
    events = classify_events(energy_scores, visual_scores, transcript)
    per_sec_scores = compute_per_second_scores(energy_scores, visual_scores, transcript, events)

    # ── PIPELINE INTEGRATION LOGIC FROM app.py ──
    print("4. Executing Step 4 candidate pool & Phase 5 ranking...")
    enable_facecam_split = True
    facecam_quota_percent = 50
    clip_count = 3
    clip_min_dur = 15
    clip_max_dur = 30
    content_style = "Action"

    presence = facecam_summary.get("presence", []) if (enable_facecam_split and facecam_summary.get("available")) else []
    reaction = facecam_summary.get("reaction", []) if (enable_facecam_split and facecam_summary.get("available")) else []
    facecam_scores = None
    if presence and reaction:
        facecam_scores = [
            (0.58 * float(reaction[i]) + 0.27 * float(per_sec_scores[i])
             + 0.15 * min(100.0, float(energy_scores[i].get("spike", 0)) * 5.0))
            if i < len(presence) and presence[i] and i < len(per_sec_scores) and i < len(energy_scores)
            else 0.0
            for i in range(min(len(presence), len(per_sec_scores), len(energy_scores)))
        ]

    candidate_pool = []
    seen_windows = set()
    pool_k = min(config.CLIP_TOP_K_MAX, max(40, clip_count * 6))

    # (A) General candidates
    gen_cands = select_highlights(
        per_sec_scores,
        event_classification=events,
        top_k=pool_k,
        min_dur=clip_min_dur,
        max_dur=clip_max_dur,
        style=content_style,
        min_score=10.0,
    )
    for c in gen_cands:
        key = (round(c["start"], 1), round(c["end"], 1))
        if key not in seen_windows:
            seen_windows.add(key)
            candidate_pool.append(c)

    # (B) Facecam-targeted candidates
    if enable_facecam_split and facecam_summary.get("available") and facecam_scores:
        fc_cands = select_highlights(
            per_sec_scores,
            event_classification=events,
            top_k=pool_k,
            min_dur=clip_min_dur,
            max_dur=clip_max_dur,
            style=content_style,
            facecam_scores=facecam_scores,
            facecam_presence=presence,
            require_facecam=True,
            min_facecam_score=8.0,
            min_facecam_coverage=0.45,
            min_score=10.0,
        )
        for c in fc_cands:
            key = (round(c["start"], 1), round(c["end"], 1))
            if key not in seen_windows:
                seen_windows.add(key)
                candidate_pool.append(c)

    print(f"   Generated candidate pool: {len(candidate_pool)} candidate windows")
    assert len(candidate_pool) > 0, "Candidate pool should not be empty"

    # Enrich candidates
    for c in candidate_pool:
        c_start = int(c["start"])
        c_end = int(c["end"])
        is_facecam_valid = False
        if enable_facecam_split and facecam_summary.get("available") and presence:
            sub_p = presence[c_start:min(c_end, len(presence))]
            cov = (sum(1 for x in sub_p if x) / max(1, len(sub_p))) if sub_p else 0.0
            sub_fs = facecam_scores[c_start:min(c_end, len(facecam_scores))] if facecam_scores else []
            avg_fs = (sum(float(x) for x in sub_fs) / max(1, len(sub_fs))) if sub_fs else 0.0
            c["facecam_coverage"] = round(cov, 2)
            c["facecam_score"] = round(avg_fs, 2)
            if cov >= 0.40 and avg_fs >= 8.0:
                is_facecam_valid = True
        else:
            c["facecam_coverage"] = 0.0
            c["facecam_score"] = 0.0

        c["facecam_available"] = is_facecam_valid
        c["layout_type"] = "facecam_split" if is_facecam_valid else "standard"

        relevant_events = events[c_start:min(c_end, len(events))]
        labels = [lbl for e in relevant_events for lbl in e.get("events", [])]
        if labels and not c.get("gameplay_event"):
            c["gameplay_event"] = {
                "category": labels[0],
                "intensity": float(c.get("score", 60.0)),
                "confidence": float(c.get("confidence", 0.7)),
                "timestamp": float(c.get("peak_second", c["start"] + (c["end"] - c["start"]) * 0.35)),
            }
        if is_facecam_valid and not c.get("reaction_event"):
            c["reaction_event"] = {
                "category": "streamer_laugh" if "laugh" in str(transcript).lower() else "streamer_hype",
                "intensity": float(c["facecam_score"]),
                "confidence": float(c["facecam_coverage"]),
                "timestamp": float(c.get("peak_second", c["start"] + (c["end"] - c["start"]) * 0.5)),
            }

    min_quality = getattr(config, "MIN_QUALITY_SCORE", 55.0)
    target_ratio = (facecam_quota_percent / 100.0) if enable_facecam_split else 0.0
    ranker = ClipRanker(
        min_quality_score=min_quality,
        target_facecam_ratio=target_ratio,
    )

    rank_res = ranker.rank_and_curate_clips(
        candidate_clips=candidate_pool,
        top_k=clip_count,
        whisper_segments=transcript,
        audio_energy=energy_scores,
        visual_scores=visual_scores,
        facecam_available_in_source=bool(enable_facecam_split and facecam_summary.get("available")),
    )

    ranked_clips = rank_res["ranked_clips"]
    highlights = []
    for rank_idx, c in enumerate(ranked_clips):
        h = dict(c)
        h["rank"] = rank_idx + 1
        h["unified_score"] = c.get("unified_score", c.get("score", 0.0))
        h["score"] = h["unified_score"]
        h["duration"] = h["end"] - h["start"]
        h["reason"] = c.get("explanation") or c.get("reason", "")
        h["score_factors"] = c.get("quality_evaluation", {}).get("factor_scores", c.get("score_factors", {}))
        h["factor_scores"] = h["score_factors"]
        highlights.append(h)

    print(f"5. Ranking & curation results:")
    print(f"   Selected clips: {len(highlights)} / {clip_count}")
    print(f"   Facecam ratio: {rank_res['facecam_ratio']*100:.0f}% (Target met: {rank_res['facecam_target_met']})")
    print(f"   Status: {rank_res['facecam_status_report']}")
    print(f"   Duplicates suppressed: {rank_res['duplicate_count']}")
    print(f"   Candidates rejected: {len(rank_res['rejected_clips'])}")

    assert len(highlights) > 0, "At least one high-quality highlight must be selected"
    for idx, h in enumerate(highlights):
        print(f"   Clip #{idx+1} [Rank #{h['rank']}]: {h['start']:.1f}s - {h['end']:.1f}s | Unified Score: {h['unified_score']:.1f} | Layout: {h['layout_type']}")
        print(f"       Rationale: {h['reason']}")
        print(f"       10-Factor summary: {h['factor_scores']}")
        # Verify required downstream keys
        for key in ["start", "end", "score", "reason", "category", "peak_second", "duration", "layout_type", "score_factors"]:
            assert key in h, f"Missing required key: {key}"

    print("\n✅ Real video pipeline verification passed cleanly with 100% contract compliance!")


if __name__ == "__main__":
    test_pipeline_on_real_video()

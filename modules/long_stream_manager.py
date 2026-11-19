"""modules/long_stream_manager.py
Orchestration for 10-12+ hour livestreams:
- Logical segmentation (>60 min into 30 min segments with boundary overlap).
- Preserves accurate global timeline timestamps across segment boundaries.
- Cross-segment candidate deduplication.
- Global candidate aggregation, scoring, and curation.
- Explainable empty-output diagnostics when no candidates pass quality gates.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.clip_ranker import ClipRanker
from modules.unified_quality_scorer import UnifiedQualityScorer

logger = logging.getLogger(__name__)


def plan_video_segments(
    total_duration_sec: float,
    segment_target_sec: float = 1800.0,
    boundary_overlap_sec: float = 30.0,
) -> list[dict]:
    """Calculate logical segment boundaries for long video processing.
    
    Videos <= 3600s (60 min) are processed as a single segment.
    Videos > 3600s are partitioned into ~30-min segments with 30s boundary overlap
    to ensure actions or reactions happening across boundaries are never lost.
    """
    if total_duration_sec <= 0:
        return []

    # Single segment if video is within 60 minutes
    if total_duration_sec <= 3600.0:
        return [{
            "segment_index": 0,
            "start_sec": 0.0,
            "end_sec": round(total_duration_sec, 2),
            "duration": round(total_duration_sec, 2),
            "is_single_segment": True,
        }]

    segments = []
    step = segment_target_sec - boundary_overlap_sec
    cur_start = 0.0
    seg_idx = 0

    while cur_start < total_duration_sec:
        cur_end = min(total_duration_sec, cur_start + segment_target_sec)
        segments.append({
            "segment_index": seg_idx,
            "start_sec": round(cur_start, 2),
            "end_sec": round(cur_end, 2),
            "duration": round(cur_end - cur_start, 2),
            "is_single_segment": False,
        })
        if cur_end >= total_duration_sec:
            break
        cur_start += step
        seg_idx += 1

    return segments


def deduplicate_cross_segment_candidates(
    candidates: list[dict],
    iou_threshold: float = 0.40,
) -> list[dict]:
    """Deduplicate candidates generated across overlapping segment boundaries.
    
    When two candidates from adjacent segments represent the same moment,
    keeps the one with higher score or larger context coverage.
    """
    if not candidates or len(candidates) <= 1:
        return list(candidates)

    # Sort primarily by start timestamp
    ordered = sorted(candidates, key=lambda c: float(c.get("start", 0.0)))
    deduped = []

    for cand in ordered:
        s = float(cand.get("start", 0.0))
        e = float(cand.get("end", 0.0))
        dur = max(0.1, e - s)

        is_duplicate = False
        for i, existing in enumerate(deduped):
            es = float(existing.get("start", 0.0))
            ee = float(existing.get("end", 0.0))
            edur = max(0.1, ee - es)

            intersection = max(0.0, min(e, ee) - max(s, es))
            union = max(0.1, max(e, ee) - min(s, es))
            iou = intersection / union
            overlap_ratio = max(intersection / dur, intersection / edur)

            if iou >= iou_threshold or overlap_ratio >= 0.50:
                is_duplicate = True
                # Keep whichever candidate has higher score / intensity
                cand_score = float(cand.get("clip_score", cand.get("unified_score", 0.0)))
                existing_score = float(existing.get("clip_score", existing.get("unified_score", 0.0)))
                if cand_score > existing_score:
                    deduped[i] = cand  # Replace with superior candidate
                break

        if not is_duplicate:
            deduped.append(cand)

    return deduped


def generate_empty_output_diagnostic(
    rank_result: dict,
    total_candidates: int,
    total_duration_sec: float,
    min_quality_score: float,
    facecam_summary: Optional[dict] = None,
) -> dict:
    """Generate rich, actionable diagnostics when 0 clips are selected.
    
    Ensures the user clearly understands WHY no clips passed and how to adjust settings.
    """
    rejection_tally = rank_result.get("rejection_tally", {})
    score_stats = rank_result.get("score_stats", {"min": 0, "max": 0, "mean": 0})
    near_misses = rank_result.get("near_misses", [])

    has_facecam = bool(facecam_summary and facecam_summary.get("detected", False))
    facecam_pct = float(facecam_summary.get("face_presence_ratio", 0.0) * 100) if facecam_summary else 0.0

    recommendations = []
    if total_candidates == 0:
        recommendations.append("No highlight candidate moments were discovered by visual/audio event detection.")
        recommendations.append("Check if video has valid audio and active gameplay movement.")
    else:
        if score_stats.get("max", 0.0) < min_quality_score:
            diff = round(min_quality_score - score_stats.get("max", 0.0), 1)
            recommendations.append(
                f"All {total_candidates} candidates scored below the current quality threshold ({min_quality_score}/100). "
                f"The highest scoring moment scored {score_stats.get('max')}/100 ({diff} pts below threshold)."
            )
            recommendations.append(
                f"Consider lowering the Minimum Quality Score in the sidebar to {max(35, int(score_stats.get('max', 50)))}."
            )

        if "Near-duplicate suppression" in rejection_tally:
            recommendations.append("Multiple candidate highlights clustered around the same moment were deduplicated.")

        if not has_facecam or facecam_pct < 20.0:
            recommendations.append(
                f"Facecam presence was low ({facecam_pct:.0f}% of stream). If this is a no-facecam or VTuber stream, "
                "ensure facecam split-screen requirement is not restricting clip selection."
            )

    return {
        "total_duration_sec": round(total_duration_sec, 1),
        "total_candidates": total_candidates,
        "accepted_count": len(rank_result.get("ranked_clips", [])),
        "min_quality_threshold": min_quality_score,
        "score_stats": score_stats,
        "rejection_tally": rejection_tally,
        "near_misses_count": len(near_misses),
        "near_misses": [
            {
                "time_range": f"{c.get('start', 0.0):.1f}s – {c.get('end', 0.0):.1f}s",
                "score": c.get("unified_score", 0.0),
                "reasons": c.get("rejection_reasons", []),
            }
            for c in near_misses
        ],
        "has_facecam": has_facecam,
        "facecam_presence_percent": round(facecam_pct, 1),
        "actionable_recommendations": recommendations,
    }

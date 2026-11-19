"""modules/clip_ranker.py
Phase 5: Intelligent Clip Ranking, Near-Duplicate Detection & Facecam Quota Balancer.

Ranks candidates best-to-worst based on explainable unified quality scores,
enforces strict context gating, eliminates near-duplicates through temporal IoU,
validates headline truthfulness, and balances the 50% facecam target ethically
without ever lowering quality thresholds or fabricating footage.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

from modules.unified_quality_scorer import UnifiedQualityScorer

logger = logging.getLogger(__name__)


class ClipRanker:
    """Intelligent clip ranking and curation system."""

    def __init__(self, target_facecam_ratio: float = 0.50, min_quality_score: float = 55.0):
        self.scorer = UnifiedQualityScorer()
        self.target_facecam_ratio = target_facecam_ratio
        self.min_quality_score = min_quality_score

    def compute_temporal_iou(self, clip1: dict[str, Any], clip2: dict[str, Any]) -> Tuple[float, float]:
        """Compute Intersection over Union (IoU) and max overlap ratio between two clips.

        Returns:
            (iou, max_overlap_ratio)
        """
        s1, e1 = float(clip1.get("start", 0)), float(clip1.get("end", 0))
        s2, e2 = float(clip2.get("start", 0)), float(clip2.get("end", 0))

        dur1 = max(0.1, e1 - s1)
        dur2 = max(0.1, e2 - s2)

        intersection = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(0.1, max(e1, e2) - min(s1, s2))

        iou = intersection / union
        max_overlap_ratio = max(intersection / dur1, intersection / dur2)
        return iou, max_overlap_ratio

    def rank_and_curate_clips(
        self,
        candidate_clips: list[dict[str, Any]],
        top_k: int = 5,
        whisper_segments: Optional[list[dict[str, Any]]] = None,
        audio_energy: Optional[list[dict[str, Any]]] = None,
        visual_scores: Optional[list[dict[str, Any]]] = None,
        facecam_available_in_source: bool = True,
    ) -> dict[str, Any]:
        """Rank, filter, deduplicate, and curate candidate clips.

        Returns:
            dict containing:
              - ranked_clips: list[dict] (best-to-worst accepted clips, up to top_k)
              - rejected_clips: list[dict] (rejected candidates with reasons)
              - duplicate_count: int
              - facecam_ratio: float (actual ratio in accepted clips)
              - facecam_target_met: bool
              - facecam_status_report: str
              - total_candidates: int
              - accepted_count: int
        """
        if not candidate_clips:
            return {
                "ranked_clips": [],
                "rejected_clips": [],
                "duplicate_count": 0,
                "facecam_ratio": 0.0,
                "facecam_target_met": False,
                "facecam_status_report": "No candidate clips provided.",
                "total_candidates": 0,
                "accepted_count": 0,
            }

        # Step 1: Initial scoring of all candidates in isolation
        scored_candidates = []
        for c in candidate_clips:
            eval_res = self.scorer.evaluate_clip(
                c,
                whisper_segments=whisper_segments,
                audio_energy=audio_energy,
                visual_scores=visual_scores,
                min_quality_score=self.min_quality_score,
            )
            annotated = dict(c)
            annotated["quality_evaluation"] = eval_res
            annotated["unified_score"] = eval_res["total_score"]
            annotated["explanation"] = eval_res["explanation"]
            annotated["rejection_reasons"] = list(eval_res["rejection_reasons"])
            scored_candidates.append(annotated)

        # Sort candidate pool initially by raw unified score descending
        scored_candidates.sort(key=lambda x: x["unified_score"], reverse=True)

        accepted_pool = []
        rejected_clips = []
        duplicate_count = 0

        # Step 2: Sequential Quality Gating & Near-Duplicate Suppression
        for cand in scored_candidates:
            # Check hard quality gate
            if (not cand["quality_evaluation"]["passed_quality_gate"]) or (cand["unified_score"] < self.min_quality_score):
                rejected_clips.append({
                    "clip": cand,
                    "reason": "; ".join(cand["rejection_reasons"]) or f"Score {cand['unified_score']:.1f} below minimum {self.min_quality_score}",
                    "stage": "quality_gate",
                })
                continue

            # Check near-duplicate against already accepted candidates
            is_dup = False
            for accepted in accepted_pool:
                iou, max_overlap = self.compute_temporal_iou(cand, accepted)
                # Overlap threshold: IoU > 0.35 or either clip covered by > 45%
                if iou > 0.35 or max_overlap > 0.45:
                    is_dup = True
                    duplicate_count += 1
                    rejected_clips.append({
                        "clip": cand,
                        "reason": f"Near-duplicate of higher-ranking clip [{accepted['start']:.1f}s - {accepted['end']:.1f}s] (IoU={iou:.2f}, overlap={max_overlap*100:.0f}%)",
                        "stage": "duplicate_suppression",
                    })
                    break

            if not is_dup:
                accepted_pool.append(cand)

        # Step 3: Quota Balancing (50% Facecam Target)
        # Partition accepted pool into facecam clips and non-facecam gameplay clips
        facecam_accepted = [c for c in accepted_pool if bool(c.get("facecam_available", False))]
        gameplay_accepted = [c for c in accepted_pool if not bool(c.get("facecam_available", False))]

        final_selected = []
        target_facecam_count = int(math.ceil(top_k * self.target_facecam_ratio))

        if not facecam_available_in_source or len(facecam_accepted) == 0:
            # No valid facecam moments exist in source footage
            final_selected = accepted_pool[:top_k]
            facecam_report = (
                f"Facecam target ({self.target_facecam_ratio*100:.0f}%) not met: source footage contains "
                f"insufficient valid facecam moments ({len(facecam_accepted)} found). "
                f"Preserved quality standard without fabricating facecam clips."
            )
            target_met = False
        else:
            # Interleave / curate top valid facecam clips and top gameplay clips
            # Select up to target_facecam_count from facecam pool
            chosen_facecam = facecam_accepted[:target_facecam_count]
            # Fill the remaining quota from gameplay pool, or remaining facecam if gameplay is insufficient
            remaining_slots = top_k - len(chosen_facecam)
            chosen_gameplay = gameplay_accepted[:remaining_slots]

            combined = chosen_facecam + chosen_gameplay
            # If we still have slots left, fill from whatever remaining high-quality clips exist
            if len(combined) < top_k:
                used_starts = {c["start"] for c in combined}
                for c in accepted_pool:
                    if c["start"] not in used_starts:
                        combined.append(c)
                        if len(combined) >= top_k:
                            break

            # Sort final selected clips by their unified quality score descending (best-to-worst ranking)
            combined.sort(key=lambda x: x["unified_score"], reverse=True)
            final_selected = combined

            facecam_count_final = sum(1 for c in final_selected if bool(c.get("facecam_available", False)))
            final_ratio = facecam_count_final / max(1, len(final_selected))
            target_met = (final_ratio >= self.target_facecam_ratio)

            if target_met:
                facecam_report = (
                    f"Facecam target met: {facecam_count_final}/{len(final_selected)} clips ({final_ratio*100:.0f}%) "
                    f"feature verified facecam split-screen."
                )
            else:
                facecam_report = (
                    f"Facecam target ({self.target_facecam_ratio*100:.0f}%) partially met: {facecam_count_final}/{len(final_selected)} "
                    f"clips ({final_ratio*100:.0f}%). Quality threshold strictly upheld without forced padding."
                )

        facecam_final_count = sum(1 for c in final_selected if bool(c.get("facecam_available", False)))
        actual_ratio = facecam_final_count / max(1, len(final_selected))

        # Build comprehensive explainability breakdown
        rejection_tally: dict[str, int] = {}
        for r in rejected_clips:
            reason_str = r.get("reason", "Unknown reason")
            # Bucket into high-level categories
            if "below minimum" in reason_str:
                cat = "Score below quality gate"
            elif "Near-duplicate" in reason_str:
                cat = "Near-duplicate suppression"
            elif "Facecam visible in only" in reason_str:
                cat = "Facecam visibility below threshold"
            elif "Missing setup" in reason_str:
                cat = "Insufficient setup context"
            elif "Abrupt cutoff" in reason_str:
                cat = "Abrupt cutoff mid-reaction"
            elif "Visual motion" in reason_str:
                cat = "Static or paused video"
            elif "silent or muted" in reason_str:
                cat = "Silent / muted audio"
            else:
                cat = reason_str[:40]
            rejection_tally[cat] = rejection_tally.get(cat, 0) + 1

        all_scores = [c["unified_score"] for c in scored_candidates] if scored_candidates else []
        score_stats = {
            "min": round(min(all_scores), 1) if all_scores else 0.0,
            "max": round(max(all_scores), 1) if all_scores else 0.0,
            "mean": round(sum(all_scores) / max(1, len(all_scores)), 1) if all_scores else 0.0,
        }

        # Identify near-miss candidates (scored just below threshold, e.g. 40.0 - 54.9)
        near_misses = [
            c for c in scored_candidates
            if not c["quality_evaluation"]["passed_quality_gate"] and c["unified_score"] >= 40.0
        ][:5]

        return {
            "ranked_clips": final_selected,
            "rejected_clips": rejected_clips,
            "duplicate_count": duplicate_count,
            "facecam_ratio": round(actual_ratio, 2),
            "facecam_target_met": (actual_ratio >= self.target_facecam_ratio),
            "facecam_status_report": facecam_report,
            "total_candidates": len(candidate_clips),
            "accepted_count": len(final_selected),
            "rejection_tally": rejection_tally,
            "score_stats": score_stats,
            "near_misses": near_misses,
        }
